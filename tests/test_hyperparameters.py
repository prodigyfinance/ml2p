# -*- coding: utf-8 -*-

""" Tests for ml2p.hyperparameters. """

import pytest

from ml2p.hyperparameters import HyperParameterEncodingError, decode, encode


class TestEncode:
    def test_int(self):
        assert encode({"a": 1}) == {"a": "1"}

    def test_float(self):
        assert encode({"a": 1.2}) == {"a": "1.2"}

    def test_str(self):
        assert encode({"a": "b"}) == {"a": '"b"'}

    def test_none(self):
        assert encode({"a": None}) == {"a": "null"}

    def test_true(self):
        assert encode({"a": True}) == {"a": "true"}

    def test_false(self):
        assert encode({"a": False}) == {"a": "false"}

    def test_list(self):
        assert encode({"a": [1, 2, 3]}) == {"a": "[1, 2, 3]"}

    def test_nested_once(self):
        assert encode({"a": {"b": 1, "c": 2}}) == {"a.b": "1", "a.c": "2"}

    def test_nested_twice(self):
        assert encode({"a": {"b1": {"c1": 3, "d1": 4}, "b2": {"c2": 5, "d2": 6}}}) == {
            "a.b1.c1": "3",
            "a.b1.d1": "4",
            "a.b2.c2": "5",
            "a.b2.d2": "6",
        }

    def test_key_format_error(self):
        with pytest.raises(HyperParameterEncodingError) as err:
            encode({"a.b": 5})
        assert str(err.value) == "Key 'a.b' must be a string without dots ('.')"

    def test_key_length_error(self):
        k = "a" * 257
        with pytest.raises(HyperParameterEncodingError) as err:
            encode({k: 5})
        assert str(err.value) == "Key '{}' must be at most 256 characters.".format(k)

    def test_nested_key_length_error(self):
        k1 = "a" * 100
        k2 = "b" * 100
        k3 = "c" * 55  # 257 minus two dots
        with pytest.raises(HyperParameterEncodingError) as err:
            encode({k1: {k2: {k3: 5}}})
        assert str(
            err.value
        ) == "Key '{}.{}.{}' must be at most 256 characters.".format(k1, k2, k3)

    def test_value_length_error(self):
        v = "b" * 255  # 257 minus two quote characters
        with pytest.raises(HyperParameterEncodingError) as err:
            encode({"a": v})
        assert str(
            err.value
        ) == "Value '\"{}\"' for key 'a' must have length at most 256.".format(v)

    def test_number_of_parameters_error(self):
        with pytest.raises(HyperParameterEncodingError) as err:
            encode({str(i): i for i in range(101)})
        assert (
            str(err.value)
            == "SageMaker allows at most 100 hyperparameters but 101 were specified."
        )


class TestDecode:
    def test_int(self):
        assert decode({"a": "1"}) == {"a": 1}

    def test_float(self):
        assert decode({"a": "1.2"}) == {"a": 1.2}

    def test_str(self):
        assert decode({"a": '"b"'}) == {"a": "b"}

    def test_none(self):
        assert decode({"a": "null"}) == {"a": None}

    def test_true(self):
        assert decode({"a": "true"}) == {"a": True}

    def test_false(self):
        assert decode({"a": "false"}) == {"a": False}

    def test_list(self):
        assert decode({"a": "[1, 2, 3]"}) == {"a": [1, 2, 3]}

    def test_nested_once(self):
        assert decode({"a.b": "1", "a.c": "2"}) == {"a": {"b": 1, "c": 2}}

    def test_nested_twice(self):
        assert decode(
            {"a.b1.c1": "3", "a.b1.d1": "4", "a.b2.c2": "5", "a.b2.d2": "6"}
        ) == {"a": {"b1": {"c1": 3, "d1": 4}, "b2": {"c2": 5, "d2": 6}}}
