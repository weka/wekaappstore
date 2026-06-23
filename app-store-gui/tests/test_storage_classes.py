from __future__ import annotations

import webapp.main as main


def test_parse_k8s_quantity_binary_suffixes():
    assert main._parse_k8s_quantity_bytes("1Ki") == 1024
    assert main._parse_k8s_quantity_bytes("10Gi") == 10 * 1024 ** 3
    assert main._parse_k8s_quantity_bytes("2Ti") == 2 * 1024 ** 4


def test_parse_k8s_quantity_decimal_suffixes():
    assert main._parse_k8s_quantity_bytes("1k") == 1000
    assert main._parse_k8s_quantity_bytes("5G") == 5 * 1000 ** 3


def test_parse_k8s_quantity_bare_and_invalid():
    assert main._parse_k8s_quantity_bytes("1024") == 1024
    assert main._parse_k8s_quantity_bytes("") == 0
    assert main._parse_k8s_quantity_bytes("notanumber") == 0
    assert main._parse_k8s_quantity_bytes(None) == 0
