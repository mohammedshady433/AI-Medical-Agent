import pytest
from app.agents.voice_facts import VoiceFacts


@pytest.mark.parametrize("source,good,bad", [
    ("لخص يومي: أشتري حاجات للبيت بعد الشغل.", "تشتري حاجات للبيت بعد العمل.", "تشتري حاجات للبيت بعد الغداء."),
    ("لخص: أتمرن قبل الغداء.", "تتمرن قبل الغداء.", "تتمرن بعد الغداء."),
    ("لخص: أشتري بعد الغداء.", "تشتري بعد الغداء.", "تشتري بعد العمل."),
    ("Recap: shop after work.", "Shop after work.", "Shop after lunch."),
    ("لخص: أتسوق بعد الشهن.", "هل توضح عبارة بعد الشهن؟", "تتسوق بعد العمل."),
    ("لخص: أشتري بعد العمل وليس بعد الغداء.", "تشتري بعد العمل، وليس بعد الغداء.", "تشتري بعد الغداء."),
])
def test_restatements_keep_relative_time_and_do_not_guess_asr(source, good, bad):
    facts = VoiceFacts(source, [])
    assert not facts.issues(good)
    assert facts.issues(bad)


def test_followup_uses_last_user_request_not_an_assistant_invention():
    history = [{"role": "user", "content": "أشتري بعد الشغل."},
               {"role": "assistant", "content": "تشتري بعد الغداء."}]
    facts = VoiceFacts("أعد تلخيص يومي.", history)
    assert facts.issues("تشتري بعد الغداء.")
    assert not facts.issues("تشتري بعد العمل.")
    correction = VoiceFacts("غير الموعد: أشتري بعد الغداء.", history)
    assert not correction.issues("تشتري بعد الغداء.")


def test_suggestions_can_introduce_new_options_and_generic_transitions_are_allowed():
    facts = VoiceFacts("اقترح وقتا للرياضة بعد العمل أو غير ذلك.", [])
    assert not facts.issues("يمكنك تجربة الرياضة قبل الفطور.")
    facts = VoiceFacts("لخص: أشتري بعد الشغل.", [])
    assert not facts.issues("تشتري بعد العمل. وبعد ذلك ترتاح.")


def test_old_unrelated_history_does_not_become_a_current_fact():
    facts = VoiceFacts("لخص الإجابة.", [{"role": "user", "content": "أتسوق بعد العمل."},
                                    {"role": "user", "content": "ما معنى الجاذبية؟"}])
    assert not facts.enabled


@pytest.mark.parametrize("equivalent", ["بعد انتهاء عملك", "عقب شغلك", "بعد الانتهاء من العمل", "بعد دوامي"])
def test_natural_possessives_and_completion_phrases_are_not_false_conflicts(equivalent):
    facts = VoiceFacts("لخص: أتسوق بعد الشغل.", [])
    assert not facts.issues(f"تتسوق {equivalent}.")
    assert facts.issues("تتسوق بعد غدائك.")
