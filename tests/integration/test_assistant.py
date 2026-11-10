import json
import pytest
from app.agents.assistant import Assistant


class ConversationProvider:
    adapter = "ollama"

    def __init__(self, config, secrets):
        self.config = config

    async def stream(self, messages, tools=()):
        assert "document-only" not in messages[0]["content"]
        yield {"type": "token", "text": "Hello. How can I help?"}


@pytest.mark.parametrize("repair,expected_check", [
    ("تشتري حاجات للبيت بعد العمل.", "repaired"),
    ("تشتري حاجات للبيت بعد الغداء.", "clarification"),
])
async def test_voice_timing_drift_is_checked_before_history_and_done(settings, secrets, database, library, repair, expected_check):
    settings.update({"language": "ar"})
    calls = []
    class DriftProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            calls.append(messages)
            yield {"type": "token", "text": "تشتري حاجات للبيت بعد الغداء." if len(calls) == 1 else repair}
    chat = database.new_chat()
    question = "لخص: أشتري حاجات للبيت بعد الشغل."
    events = [e async for e in Assistant(settings, secrets, database, library, DriftProvider).reply(chat, question, voice=True)]
    final = events[-1]
    assert len(calls) == 2
    assert final["meta"]["voice_fact_check"] == expected_check
    assert final["meta"]["model"] == "gpt-oss:20b"
    assert "الغداء" not in final["content"]
    assert any(e["type"] == "status" and e["tool"] == "verify_voice_facts" for e in events)
    assert database.messages(chat)[0]["content"] == question
    assert database.messages(chat)[1]["content"] == final["content"]


async def test_accurate_voice_reply_requires_no_second_model_call(settings, secrets, database, library):
    calls = []
    class AccurateProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            calls.append(messages)
            yield {"type": "token", "text": "تشتري بعد العمل."}
    events = [e async for e in Assistant(settings, secrets, database, library, AccurateProvider).reply(database.new_chat(), "لخص: أشتري بعد الشغل.", voice=True)]
    assert len(calls) == 1
    assert events[-1]["content"] == "تشتري بعد العمل."


async def test_voice_prepare_uses_same_prompt_without_saved_conversation(settings, secrets, database, library):
    from app.agents.assistant import voice_config
    captured = {}
    class WarmProvider(ConversationProvider):
        async def preload(self, messages, tools):
            captured.update(messages=messages, tools=tools)
    database.add_message(database.new_chat(), "user", "PRIVATE HISTORY SENTINEL")
    assistant = Assistant(settings, secrets, database, library, WarmProvider)
    cfg = voice_config(settings.snapshot())
    await assistant.prepare_voice(cfg)
    assert captured["messages"][0]["content"] == assistant.personality(cfg, True)
    assert captured["messages"][1] == {"role": "user", "content": ""}
    assert "PRIVATE HISTORY SENTINEL" not in str(captured)


async def test_general_conversation_and_history(settings, secrets, database, library):
    chat = database.new_chat()
    assistant = Assistant(settings, secrets, database, library, ConversationProvider)
    result = [e async for e in assistant.reply(chat, "Hi")]
    assert result[-1]["type"] == "done"
    assert [m["role"] for m in database.messages(chat)] == ["user", "assistant"]
    settings.update({"mode": "offline"})
    assert len(database.messages(chat)) == 2


class SourceProvider(ConversationProvider):
    async def stream(self, messages, tools=()):
        if messages[-1]["role"] != "tool":
            yield {"type": "tool_call", "call": {"function": {"name": "document_overview", "arguments": {}}}}
        else:
            evidence = json.loads(messages[-1]["content"])
            ref = evidence["sources"][0]["reference"]
            yield {
                "type": "token",
                "text": "ميزانية التدريب 8000 جنيه "
                + ref.replace("[", "【").replace("]", "】")
                + " 【D999999】",
            }


async def test_tool_planning_is_not_saved_or_spoken_as_the_final_answer(settings, secrets, database, library):
    class ToolProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            if messages[-1]["role"] != "tool":
                yield {"type": "token", "text": "An unverified guess. "}
                yield {"type": "tool_call", "call": {"function": {"name": "calculator", "arguments": {"expression": "2+3"}}}}
            else:
                yield {"type": "token", "text": "Five."}
    chat = database.new_chat()
    events = [e async for e in Assistant(settings, secrets, database, library, ToolProvider).reply(chat, "2+3?", voice=True)]
    assert events[-1]["content"] == "Five."
    assert database.messages(chat)[-1]["content"] == "Five."


async def test_voice_turn_is_concise_without_changing_saved_budget(settings, secrets, database, library):
    original = settings.snapshot()["models"]["online"]["max_response_tokens"]

    class VoiceProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            assert self.config["mode"] == "offline"
            assert self.config["models"]["offline"]["max_response_tokens"] <= 512
            assert self.config["models"]["online"]["max_response_tokens"] == original
            assert "spoken conversation" in messages[0]["content"]
            yield {"type": "token", "text": "A short spoken answer."}

    result = [
        event
        async for event in Assistant(settings, secrets, database, library, VoiceProvider).reply(
            database.new_chat(), "Hello", voice=True
        )
    ]
    assert result[-1]["type"] == "done"
    assert settings.snapshot()["models"]["online"]["max_response_tokens"] == original


async def test_arabic_answer_english_evidence_invalid_reference_safe(
    settings, secrets, database, library, tmp_path
):
    settings.update({"mode": "offline"})
    with database.connect() as db:
        db.execute(
            "INSERT INTO documents(id,name,digest,status,created) VALUES('a','budget.txt','h','ready',1)"
        )
        db.execute(
            "INSERT INTO chunks(document_id,location,text,normalized) VALUES('a','Paragraph 1','Training budget is 8000 EGP.','training budget is 8000 egp')"
        )
    chat = database.new_chat()
    assistant = Assistant(settings, secrets, database, library, SourceProvider)
    result = [e async for e in assistant.reply(chat, "إيه في الملفات؟")]
    assert result[-1]["type"] == "done"
    assert "[D999999]" not in result[-1]["content"]
    assert result[-1]["meta"]["sources"][0]["name"] == "budget.txt"


async def test_prior_document_answers_not_sent_when_cloud_privacy_off(settings, secrets, database, library):
    chat = database.new_chat()
    database.add_message(chat, "assistant", "PRIVATE FILE EXCERPT", {"document_context": True})

    class PrivacyProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            assert "PRIVATE FILE EXCERPT" not in json.dumps(messages)
            assert "search_documents" not in json.dumps(tools)
            yield {"type": "token", "text": "Hi"}

    result = [
        e async for e in Assistant(settings, secrets, database, library, PrivacyProvider).reply(chat, "Hi")
    ]
    assert result[-1]["type"] == "done"


async def test_tool_injection_cannot_call_unregistered_tools(settings, secrets, database, library):
    class UnsafeProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            if messages[-1]["role"] == "tool":
                assert "error" in json.loads(messages[-1]["content"])
                yield {"type": "token", "text": "That tool is unavailable."}
            else:
                yield {
                    "type": "tool_call",
                    "call": {"function": {"name": "run_shell", "arguments": {"command": "echo BAD"}}},
                }

    result = [
        e
        async for e in Assistant(settings, secrets, database, library, UnsafeProvider).reply(
            database.new_chat(), "hello"
        )
    ]
    assert result[-1]["type"] == "done"


async def test_voice_preserves_transcript_instead_of_silently_guessing_a_phrase(settings, secrets, database, library):
    settings.update({"language": "ar"})
    seen = []

    class CaptureProvider(ConversationProvider):
        async def stream(self, messages, tools=()):
            seen.append(messages[-1]["content"])
            yield {"type": "token", "text": "أقدر أساعدك في حاجات تانية."}

    assistant = Assistant(settings, secrets, database, library, CaptureProvider)
    chat = database.new_chat()
    database.add_message(chat, "assistant", "أقدر أساعدك في الترجمة.")
    for question, voice in [("أوي تين", True), ("أوي تين", False), ("يعني إيه أوي تين؟", True)]:
        async for _ in assistant.reply(chat, question, voice=voice):
            pass
    assert seen == ["أوي تين", "أوي تين", "يعني إيه أوي تين؟"]
    assert database.messages(chat)[1]["content"] == "أوي تين"

async def test_unclear_spoken_clock_keeps_the_whole_request_and_asks_before_model(settings, secrets, database, library):
    settings.update({'mode': 'offline', 'language': 'ar'})
    def forbidden(*args):
        raise AssertionError('An unclear clock hour must be clarified before generation.')
    assistant = Assistant(settings, secrets, database, library, forbidden)
    for clock in ['الساعة ساعة الصبح', 'الساعة الساعة الصبح', 'السَّاعة الصَّباح']:
        chat = database.new_chat()
        question = f'عندي اجتماع {clock} واشتري حاجات للبيت وعندي موعد الساعة عشرة بالليل'
        result = [e async for e in assistant.reply(chat, question, voice=True)]
        assert result[-1]['meta']['model'] == 'speech-clarification'
        assert 'ما الساعة المقصودة؟' in result[-1]['content']
        assert '1' not in result[-1]['content'] and 'الواحدة' not in result[-1]['content']
        assert database.messages(chat)[0]['content'] == question
        class FollowUp(ConversationProvider):
            async def stream(self, messages, tools=()):
                assert question in [m['content'] for m in messages]
                assert 'حاجات للبيت' in str(messages) and 'عشرة بالليل' in str(messages)
                yield {'type': 'token', 'text': 'موعد الاجتماع التاسعة صباحاً.'}
        followup = Assistant(settings, secrets, database, library, FollowUp)
        assert [e async for e in followup.reply(chat, 'الساعة تسعة الصبح', voice=True)][-1]['type'] == 'done'

async def test_clear_spoken_hours_and_typed_questions_still_use_selected_model(settings, secrets, database, library):
    settings.update({'mode': 'offline', 'language': 'ar'})
    assistant = Assistant(settings, secrets, database, library, ConversationProvider)
    for question, voice in [('الساعة تسعة الصبح', True), ('الساعة 9 صباحا', True), ('الساعة ساعة الصبح', False)]:
        result = [e async for e in assistant.reply(database.new_chat(), question, voice=voice)]
        assert result[-1]['content'] == 'Hello. How can I help?'
        assert result[-1]['meta']['model'] != 'speech-clarification'
