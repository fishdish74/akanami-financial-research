"""Exercise the actual urllib opener offline, intercepting only HTTPS sends."""
from __future__ import annotations

import io
import ssl
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request
from urllib.response import addinfourl

from market_analytics.massive_provider import MassiveValidationProvider, _default_transport
from market_analytics.provider_runtime import ProviderBoundaryError, ProviderRequestError


SOURCE = "https://api.massive.com/v3/reference/tickers/AAPL"
TOKEN = "synthetic-offline-credential"


class MassiveRedirectTests(unittest.TestCase):
    def fixture(self, statuses, *, location=None, body=b'{"status":"OK","results":[]}'):
        self.sends = []
        self.contexts = []
        self.sleeps = []
        def send(handler, request):
            self.sends.append((request.full_url, dict(request.header_items()), request.timeout))
            self.contexts.append(handler._context)
            status = statuses[min(len(self.sends) - 1, len(statuses) - 1)]
            if isinstance(status, Exception):
                raise status
            headers = Message()
            if location:
                headers['Location'] = location
            response = addinfourl(io.BytesIO(body), headers, request.full_url, status)
            response.msg = 'synthetic response'
            return response
        return patch.object(HTTPSHandler, 'https_open', send)

    def provider(self):
        return MassiveValidationProvider(TOKEN, clock=lambda: 0., sleeper=self.sleeps.append,
                                        requests_per_minute=100)

    def check_redirect(self, status):
        for destination in ('https://unrelated.invalid/receive',
                            'https://api.massive.com/redirected'):
            with self.subTest(status=status, destination=destination):
                with self.fixture([status], location=destination + '?opaque=' + TOKEN,
                                  body=(TOKEN * 1000).encode()):
                    provider = self.provider()
                    with self.assertRaises(ProviderRequestError) as caught:
                        provider._request_json(SOURCE)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(str(caught.exception), 'unexpected_redirect')
                self.assertNotIn(TOKEN, str(caught.exception))
                self.assertEqual(len(self.sends), 1)
                self.assertEqual(self.sends[0][0], SOURCE)
                self.assertEqual(self.sends[0][1]['Authorization'], 'Bearer ' + TOKEN)
                self.assertEqual(provider.usage.requests_attempted, 1)
                self.assertEqual(provider.usage.requests_failed, 1)
                self.assertEqual(provider.usage.requests_succeeded, 0)
                self.assertEqual(provider.usage.estimated_billable_calls, 1)
                self.assertEqual(provider.usage.retries, 0)
                self.assertEqual(self.sleeps, [])

    def test_301(self): self.check_redirect(301)
    def test_302(self): self.check_redirect(302)
    def test_303(self): self.check_redirect(303)
    def test_307(self): self.check_redirect(307)
    def test_308(self): self.check_redirect(308)

    def test_positive_control_default_handler_would_forward_authorization(self):
        # No network: proves the stdlib behavior the custom opener must override.
        request = Request(SOURCE, headers={'Authorization': 'Bearer ' + TOKEN})
        redirected = HTTPRedirectHandler().redirect_request(
            request, None, 302, 'Found', {}, 'https://unrelated.invalid/receive')
        self.assertEqual(redirected.get_header('Authorization'), 'Bearer ' + TOKEN)

    def test_transport_discards_redirect_location_and_body(self):
        with self.fixture([302], location='https://unrelated.invalid/?key=' + TOKEN,
                          body=TOKEN.encode()):
            result = _default_transport(SOURCE, {'Authorization': 'Bearer ' + TOKEN}, 15., 1)
        self.assertEqual((result.status, result.body, result.headers), (302, b'', {}))
        self.assertEqual(len(self.sends), 1)

    def test_malformed_or_missing_location_is_not_retried(self):
        for location in (None, 'https://[invalid', 'file:///not-a-network-endpoint'):
            with self.subTest(location=location), self.fixture([302], location=location):
                provider = self.provider()
                with self.assertRaisesRegex(ProviderRequestError, '^unexpected_redirect$'):
                    provider._request_json(SOURCE)
                self.assertEqual(len(self.sends), 1)
                self.assertEqual(provider.usage.retries, 0)

    def test_normal_200_parsing_timeout_and_tls_defaults_preserved(self):
        with self.fixture([200]):
            provider = self.provider()
            payload, _ = provider._request_json(SOURCE)
        self.assertEqual(payload, {'status': 'OK', 'results': []})
        self.assertEqual(len(self.sends), 1)
        self.assertEqual(self.sends[0][2], 15.)
        self.assertEqual(self.sends[0][1]['Accept-encoding'], 'identity')
        for context in self.contexts:
            # None delegates to HTTPSConnection's standard verified TLS context.
            context = context or ssl._create_default_https_context()
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
        self.assertEqual(provider.controller.policy.max_retries, 2)
        self.assertEqual(provider.controller.policy.daily_quota, 20)
        self.assertEqual(provider._max_pages, 3)

    def test_401_and_403_remain_nonretryable(self):
        for status in (401, 403):
            with self.subTest(status=status), self.fixture([status]):
                provider = self.provider()
                with self.assertRaises(ProviderRequestError) as caught:
                    provider._request_json(SOURCE)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(len(self.sends), 1)
                self.assertEqual(provider.usage.retries, 0)

    def test_429_and_5xx_keep_bounded_retry_behavior(self):
        for status in (429, 500):
            with self.subTest(status=status), self.fixture([status, 200]):
                provider = self.provider()
                provider._request_json(SOURCE)
                self.assertEqual(len(self.sends), 2)
                self.assertEqual(provider.usage.requests_attempted, 2)
                self.assertEqual(provider.usage.retries, 1)

    def test_timeout_still_stops_after_three_attempts(self):
        with self.fixture([TimeoutError('synthetic timeout')]):
            provider = self.provider()
            with self.assertRaises(ProviderRequestError):
                provider._request_json(SOURCE)
        self.assertEqual(len(self.sends), 3)
        self.assertEqual(provider.usage.retries, 2)

    def test_malformed_json_is_not_success(self):
        with self.fixture([200], body=b'not JSON'):
            provider = self.provider()
            with self.assertRaises(ProviderBoundaryError):
                provider._request_json(SOURCE)
        self.assertEqual(len(self.sends), 1)

    def test_normal_response_size_limit_preserved(self):
        with self.fixture([200], body=b'large'):
            with self.assertRaisesRegex(ProviderBoundaryError, 'size limit'):
                _default_transport(SOURCE, {}, 15., 1)
        self.assertEqual(len(self.sends), 1)


if __name__ == '__main__':
    unittest.main()
