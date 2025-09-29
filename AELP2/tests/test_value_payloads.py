#!/usr/bin/env python3
from AELP2.adapters.google_enhanced_conversions import build_enhanced_conversions
from AELP2.adapters.meta_capi import build_capi_events


def test_google_ec_payload():
    rows=[{'email':'User@Example.com','phone':'(555) 123-4567','first_name':'Ann','last_name':'Lee','country':'us','postal_code':'02139','value':123.45,'currency':'USD','conversion_action':'123','order_id':'o1'}]
    out=build_enhanced_conversions(rows)
    assert out and out[0]['conversion_action']=='123' and 'user_identifiers' in out[0]


def test_meta_capi_payload():
    rows=[{'email':'u@e.com','phone':'555-999-0000','event_name':'Purchase','event_time':1700000000,'value':50,'currency':'USD'}]
    out=build_capi_events(rows)
    assert out and out[0]['event_name']=='Purchase' and 'user_data' in out[0]


if __name__=='__main__':
    test_google_ec_payload(); test_meta_capi_payload(); print('value payload tests OK')

