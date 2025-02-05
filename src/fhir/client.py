from fhirclient import client
from config.settings import Settings


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
            patient = self.smart.server.get_patient(patient_id)
            return patient.as_json()
        except Exception as e:
            print(f"Error fetching patient: {e}")
            return None
