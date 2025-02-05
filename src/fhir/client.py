from src.config.settings import Settings
from fhirclient import client
from fhirclient.models import patient


class FHIRClient:
    def __init__(self):
        self.settings = Settings()
        self.smart = client.FHIRClient(
            settings={
                "app_id": "ehr_adaptive_agent",
                "api_base": self.settings.FHIR_SERVER_URL,
            }
        )

    def get_patient(self, patient_id: str):
        """Fetch a patient resource by ID"""
        try:
            search = patient.Patient.read(patient_id, self.smart.server)
            return search.as_json()
        except Exception as e:
            print(f"Error fetching patient: {e}")
            return None
