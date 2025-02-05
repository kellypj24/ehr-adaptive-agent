import pytest
from src.fhir.explorer import FHIRResourceExplorer

@pytest.fixture
def explorer():
    return FHIRResourceExplorer()

@pytest.mark.asyncio
async def test_explore_patient_structure(explorer):
    result = await explorer.explore_resource_structure("Patient")
    assert result is not None
    assert "structure" in result
    assert "resourceType" in result["structure"]

@pytest.mark.asyncio
async def test_get_patient_relationships(explorer):
    result = await explorer.get_resource_relationships("example", "Patient")
    assert result is not None
    assert "references" in result
    assert isinstance(result["references"], list)