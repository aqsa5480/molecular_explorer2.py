Molecular Explorer Pro | by Aqsa Ijaz 🧪

Interactive chemistry visualization & analysis tool. Explore molecules, calculate properties, optimize geometry, and interact with a chemistry bot.

🚀 Features

Quick analysis of molecular formula, weight, and elements

Detailed compound properties (PubChem integration)

3D molecular visualization with Stick, Sphere, Cartoon, Surface styles

Quantum/geometry optimization (local ASE-enabled only)

Chemistry Q&A Bot (text & optional voice input)

Lipinski rule evaluation, bioavailability estimate, SAScore

Cached queries for faster performance

💻Installation
1. Clone the repo
git clone https://github.com/<your-username>/MolecularExplorerPro.git
cd MolecularExplorerPro

2. Create environment (recommended)
conda create -n molexplorer python=3.11
conda activate molexplorer

3. Install dependencies
pip install -r requirements.txt


⚠️ Note: ASE and voice assistant features only work locally.

🌐 Run the App
streamlit run main.py


Access in browser at http://localhost:8501

Voice input works only on local machine

3D visualizations supported both locally and in cloud

🧪 Usage

Select Analysis Mode from sidebar: Quick, Detailed, 3D Explorer, Quantum Calc

Search or select a compound (local database or PubChem)

View molecular properties, generate 3D structures

Ask chemistry questions via text or voice input

⚡ Optional Features

Voice Assistant: Requires microphone & speaker (local)

Geometry Optimization: ASE-based, works locally

PubChem Fetch: Retrieves compound info dynamically
