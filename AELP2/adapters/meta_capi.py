import hashlib
from typing import Dict, Any, List


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def build_capi_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build Meta CAPI events payload with hashed PII.

    Input fields: event_name, event_time, value, currency, email, phone, fbp, fbc, client_ip, client_user_agent.
    """
    out = []
    for r in rows:
        email = normalize_email(str(r.get('email','')))
        phone = ''.join(filter(str.isdigit, str(r.get('phone',''))))
        user_data = {
            'em': _sha256(email) if email else None,
            'ph': _sha256(phone) if phone else None,
            'client_ip_address': r.get('client_ip'),
            'client_user_agent': r.get('client_user_agent'),
            'fbp': r.get('fbp'),
            'fbc': r.get('fbc'),
        }
        event = {
            'event_name': r.get('event_name') or 'Purchase',
            'event_time': int(r.get('event_time') or 0),
            'event_source_url': r.get('event_source_url'),
            'action_source': r.get('action_source') or 'website',
            'custom_data': {
                'currency': r.get('currency') or 'USD',
                'value': float(r.get('value') or 0.0),
                'order_id': r.get('order_id'),
            },
            'user_data': {k:v for k,v in user_data.items() if v},
        }
        out.append(event)
    return out

