import asyncio
from src.fhir.explorer import FHIRResourceExplorer

async def main():
    explorer = FHIRResourceExplorer()
    
    print("Starting FHIR resource exploration...")
    
    # Explore Patient resource structure
    print("\nExploring Patient resource structure:")
    patient_structure = await explorer.explore_resource_structure("Patient")
    print("\nPatient Structure:", patient_structure)
    
    # Explore relationships
    print("\nExploring Patient relationships:")
    relationships = await explorer.get_resource_relationships("example", "Patient")
    print("\nRelationships:", relationships)

if __name__ == "__main__":
    asyncio.run(main())