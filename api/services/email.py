"""
api/services/email.py

Sends the magic-link email. Uses the Resend HTTP API (a single POST, no SDK
dependency needed beyond `requests`, which is already a project dependency).

When EMAIL_PROVIDER_API_KEY is unset (local dev, or before the provider is
configured on Railway), the link is logged instead of sent so the auth flow
is still testable end-to-end without email infrastructure.

This must never raise: a failed send should not reveal to the caller whether
an account exists, and request-link's response is identical either way.
"""

from __future__ import annotations

import os

import requests


def send_magic_link_email(to_email: str, link: str) -> bool:
    api_key = os.getenv("EMAIL_PROVIDER_API_KEY")
    from_addr = os.getenv("EMAIL_FROM", "Macro Engine <hello@macro-engine.com>")
    subject = "Your Macro Engine sign-in link"

    text_body = (
        "Click the link below to sign in to Macro Engine.\n\n"
        "{link}\n\n"
        "This link expires in 15 minutes and can only be used once. "
        "If you didn't request it, you can ignore this email."
    ).format(link=link)

    html_body = (
        "<p>Click the link below to sign in to Macro Engine.</p>"
        '<p><a href="{link}">{link}</a></p>'
        "<p>This link expires in 15 minutes and can only be used once. "
        "If you didn't request it, you can ignore this email.</p>"
    ).format(link=link)

    if not api_key:
        print("[auth] EMAIL_PROVIDER_API_KEY not set; magic link for {}: {}".format(to_email, link))
        return False

    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": "Bearer {}".format(api_key)},
            json={
                "from": from_addr,
                "to": [to_email],
                "subject": subject,
                "text": text_body,
                "html": html_body,
            },
            timeout=10,
        )
        if resp.status_code >= 300:
            print("[auth] email send failed ({}): {}".format(resp.status_code, resp.text[:300]))
            return False
        return True
    except Exception as e:
        print("[auth] email send error: {}".format(e))
        return False
