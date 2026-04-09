import chromadb
from chromadb.utils import embedding_functions

# Use a lightweight local embedding model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./clinical_kb")
collection = client.get_or_create_collection(
    name="parkinson_guidelines",
    embedding_function=ef
)


def build_knowledge_base():
    """Load clinical knowledge into the vector store."""
    
    documents = [
        # MDS-UPDRS Grade definitions
        "MDS-UPDRS Item 3.3 Grade 0: No increase in muscle tone detected during passive movement of major joints at rest.",
        "MDS-UPDRS Item 3.3 Grade 1: Slight increase in muscle tone, manifested as a catch and release or minimal resistance at the end of range of motion during passive movement.",
        "MDS-UPDRS Item 3.3 Grade 2: Mild increase in muscle tone, manifested as a catch followed by minimal resistance throughout the remainder of passive range of motion.",
        "MDS-UPDRS Item 3.3 Grade 3: Moderate increase in muscle tone through most of range of motion, but affected parts easily moved.",
        
        # Clinical observations by grade
        "Patients with Grade 1 Parkinson's rigidity may show slight resistance only when the examiner activates the limb. Walking speed may be near normal. Arm swing may be slightly reduced on the affected side.",
        "Patients with Grade 2 Parkinson's rigidity typically show reduced arm swing, slightly stooped posture, and mildly slow gait. Resistance is detectable without activation maneuver.",
        "Patients with Grade 3 Parkinson's rigidity show markedly reduced or absent arm swing, stooped posture, shuffling gait, and significant resistance throughout range of motion.",
        
        # Referral guidelines
        "Patients with new onset rigidity Grade 2 or above should be referred to a neurologist or movement disorder specialist within 4 weeks.",
        "Patients with Grade 3 rigidity, falls history, or rapid progression should be referred urgently within 1 week.",
        "Grade 1 rigidity with no functional impairment may be monitored by a community health worker with 3-month follow-up.",
        
        # Differential considerations
        "Rigidity must be distinguished from spasticity. Rigidity in Parkinson's disease is present throughout passive range of motion (lead-pipe) or with superimposed tremor (cogwheel). Spasticity is velocity-dependent.",
        "Cogwheel rigidity, a ratchet-like resistance during passive movement, is characteristic of Parkinson's disease and indicates the presence of both rigidity and tremor.",
        "Rigidity in Parkinson's disease typically begins asymmetrically, affecting one side before the other, which distinguishes it from other parkinsonian syndromes.",
        
        # Medication context
        "Parkinson's rigidity severity can fluctuate significantly with medication timing. Assessments should note whether the patient is in an 'on' state (medication effective) or 'off' state (medication wearing off).",
        "Levodopa is the most effective medication for Parkinson's rigidity. Patients on stable levodopa therapy may show significantly reduced rigidity compared to unmedicated baseline.",
        
        # Community health worker guidance
        "Community health workers should observe gait, arm swing, facial expression, and posture during a 10-metre walk test to inform rigidity assessment.",
        "In low-resource settings, the timed up and go test (TUG) is a practical screening tool. A TUG time greater than 12 seconds indicates significant motor impairment.",
        "Health workers should ask about medication timing, falls in the past month, and whether symptoms have worsened since last assessment.",
    ]
    
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Only add if collection is empty
    if collection.count() == 0:
        collection.add(documents=documents, ids=ids)
        print(f"Knowledge base built with {len(documents)} documents")
    else:
        print(f"Knowledge base already exists with {collection.count()} documents")


def retrieve_relevant_context(query, n_results=4):
    """Retrieve most relevant clinical guidelines for a given query."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    if results and results["documents"]:
        passages = results["documents"][0]
        return "\n\n".join(f"• {p}" for p in passages)
    
    return "No specific guidelines retrieved."