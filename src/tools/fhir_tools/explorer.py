from typing import Dict, List, Optional
from fhirclient import client
from .client import FHIRClient
from fhirclient.models import patient


class FHIRExplorer:
    def __init__(self):
        self.client = FHIRClient()
        self.resource_types = [
            "Patient",
            "Observation",
            "Condition",
            "Procedure",
            "MedicationRequest",
        ]

    async def explore_resource_structure(self, resource_type: str) -> Dict:
        """Explore the structure of a FHIR resource type"""
        try:
            if resource_type == "Patient":
                # Get a sample patient to explore structure
                sample = patient.Patient.read("example", self.client.smart.server)
                structure = {
                    "resourceType": "Patient",
                    "fields": list(sample.as_json().keys()),
                    "sample": sample.as_json(),
                }
                return {"structure": structure}
            return {"error": f"Resource type {resource_type} not yet supported"}
        except Exception as e:
            return {"error": f"Error exploring {resource_type}: {str(e)}"}

    def _analyze_structure(self, resource: Dict) -> Dict:
        """Recursively analyzes the structure of a FHIR resource"""
        structure = {}

        for key, value in resource.items():
            if isinstance(value, dict):
                structure[key] = {
                    "type": "object",
                    "properties": self._analyze_structure(value),
                }
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        structure[key] = {
                            "type": "array",
                            "items": self._analyze_structure(value[0]),
                        }
                    else:
                        structure[key] = {
                            "type": "array",
                            "items": {"type": type(value[0]).__name__},
                        }
                else:
                    structure[key] = {"type": "array", "items": {}}
            else:
                structure[key] = {"type": type(value).__name__}

        return structure

    async def get_resource_relationships(
        self, resource_id: str, resource_type: str
    ) -> Dict:
        """Get relationships for a specific resource"""
        try:
            if resource_type == "Patient":
                # Get the patient resource
                pat = patient.Patient.read(resource_id, self.client.smart.server)
                pat_json = pat.as_json()

                # Extract references from the patient resource
                references = []
                # Look for common reference fields in Patient resource
                if "managingOrganization" in pat_json:
                    references.append(
                        {
                            "type": "Organization",
                            "field": "managingOrganization",
                            "reference": pat_json["managingOrganization"],
                        }
                    )
                if "generalPractitioner" in pat_json:
                    references.append(
                        {
                            "type": "Practitioner",
                            "field": "generalPractitioner",
                            "reference": pat_json["generalPractitioner"],
                        }
                    )

                return {"references": references}
            return {"error": f"Resource type {resource_type} not yet supported"}
        except Exception as e:
            return {"error": f"Error exploring relationships: {str(e)}"}

    def _find_references(
        self, data: Dict, references: List[str], path: str = ""
    ) -> None:
        """Recursively finds all references in a FHIR resource"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key

                if key == "reference" and isinstance(value, str):
                    references.append({"path": path, "reference": value})
                elif isinstance(value, (dict, list)):
                    self._find_references(value, references, new_path)
        elif isinstance(data, list):
            for item in data:
                self._find_references(item, references, path)
