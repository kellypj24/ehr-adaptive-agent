import pytest
from src.fhir.client import FHIRClient


def test_fhir_client_initialization():
    client = FHIRClient()
    assert client.settings.FHIR_SERVER_URL == "https://hapi.fhir.org/baseR4"


def test_get_patient():
    client = FHIRClient()
    # Using a known test patient ID from HAPI FHIR
    patient = client.get_patient("example")
    assert patient is not None
    assert "resourceType" in patient
    assert patient["resourceType"] == "Patient"


def test_get_invalid_patient():
    client = FHIRClient()
    # Test with a non-existent patient ID
    patient = client.get_patient("nonexistent-123")
    assert patient is None
