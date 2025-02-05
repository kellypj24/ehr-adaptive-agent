import pytest
from fhir.client import FHIRClient


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
