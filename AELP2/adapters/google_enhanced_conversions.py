import hashlib
import json
from typing import Dict, Any, List


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def build_enhanced_conversions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build Google Enhanced Conversions payloads with SHA256-hashed PII.

    Input rows must include: email, phone, first_name, last_name, country, postal_code, value, currency, conversion_action.
    """
    out = []
    for r in rows:
        email = normalize_email(str(r.get('email','')))
        phone = ''.join(filter(str.isdigit, str(r.get('phone',''))))
        fname = (r.get('first_name') or '').strip().lower()
        lname = (r.get('last_name') or '').strip().lower()
        country = (r.get('country') or '').strip().upper()
        postal = (r.get('postal_code') or '').strip().upper()
        user_ids = {
            'hashed_email': _sha256(email) if email else None,
            'hashed_phone_number': _sha256(phone) if phone else None,
            'address_info': {
                'hashed_first_name': _sha256(fname) if fname else None,
                'hashed_last_name': _sha256(lname) if lname else None,
                'country_code': country or None,
                'postal_code': postal or None,
            },
        }
        payload = {
            'conversion_action': r.get('conversion_action'),
            'conversion_value': float(r.get('value') or 0.0),
            'currency_code': r.get('currency') or 'USD',
            'user_identifiers': {k:v for k,v in user_ids.items() if v},
            'order_id': r.get('order_id'),
        }
        out.append(payload)
    return out

