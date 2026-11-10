"""Public HTTPS retrieval with address pinning and bounded response bodies."""
import asyncio
import http.client
from html.parser import HTMLParser
import ipaddress
import socket
import ssl
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx


def public_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443) or len(url) > 2000:
        raise ValueError("Research only fetches public HTTPS pages on port 443.")
    if any(ord(c) < 33 for c in url):
        raise ValueError("Invalid website address.")
    if parsed.hostname.casefold() in {"localhost", "localhost.localdomain"} or parsed.hostname.endswith(".local"):
        raise ValueError("Local network pages are not research sources.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Private addresses are not research sources.")
    return parsed


class PublicConnection(http.client.HTTPSConnection):
    def connect(self):
        addresses = socket.getaddrinfo(self.host, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
            raise ValueError("The website resolves to a private or reserved network address.")
        # Connect to the exact validated address while retaining TLS hostname verification.
        family, kind, proto, _, address = addresses[0]
        sock = socket.socket(family, kind, proto)
        sock.settimeout(self.timeout)
        try:
            sock.connect(address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


def fetch_html(url, limit=750_000):
    for _ in range(4):
        parsed = public_url(url)
        connection = PublicConnection(parsed.hostname, timeout=12, context=ssl.create_default_context())
        try:
            target = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
            connection.request("GET", target, headers={"User-Agent": "EMS-Research/1.1", "Accept": "text/html,text/plain", "Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.getheader("Location", ""))
                continue
            if response.status != 200 or not any(t in response.getheader("Content-Type", "") for t in ("text/html", "text/plain")):
                raise ValueError("The source did not return readable text.")
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ValueError("Source page exceeds the retrieval limit.")
            return body.decode("utf8", errors="replace"), url
        finally:
            connection.close()
    raise ValueError("The source redirected too many times.")


class PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}: self.skip += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}: self.skip = max(0, self.skip - 1)
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip())


class SearchResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results, self.current = [], None
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "result__a" in values.get("class", "").split():
            url = values.get("href", "")
            if url.startswith("//"): url = "https:" + url
            parsed = urlsplit(url)
            if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
                url = parse_qs(parsed.query).get("uddg", [url])[0]
            self.current = {"url": url, "title": ""}
    def handle_data(self, data):
        if self.current is not None: self.current["title"] += data
    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            try:
                public_url(self.current["url"])
                self.results.append(self.current)
            except ValueError:
                pass
            self.current = None


class WebSources:
    async def search(self, query, count=5):
        # Fixed search endpoint; queries never become arbitrary fetch targets.
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            async with client.stream("GET", "https://html.duckduckgo.com/html/", params={"q": query}, headers={"User-Agent": "EMS-Research/1.1"}) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 750_000: raise ValueError("Search response too large.")
        parser = SearchResults(); parser.feed(body.decode("utf8", errors="replace"))
        return parser.results[:count]

    async def page(self, url, max_chars):
        html, final_url = await asyncio.to_thread(fetch_html, url)
        parser = PageText(); parser.feed(html)
        return {"url": final_url, "text": "\n".join(parser.parts)[:max_chars]}
