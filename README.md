==================================================================
Aalo LLM Gateway
==================================================================
A high-performance FastAPI gateway powering the Aalo application. This service acts as a secure, stateless middleware that connects a public-facing frontend to a localized LLM model using Cloudflare Zero Trust Tunnels.

Developed to serve as the backend for "Alo"—a custom AI persona functioning as an empathetic yet highly analytical technical operations partner.

==================================================================
Architecture & Features
==================================================================
⚬ Stateless Proxy: Routes chat requests to local LLM server with support for both real-time Server-Sent Events (SSE) streaming and standard JSON responses.
⚬ Dynamic System Prompt: Enforces the core system prompt dynamically at the API layer via pydantic_settings, preventing client-side prompt overrides.
⚬ Secure Exfiltration: Operates behind a persistent, token-based Cloudflare Tunnel (cloudflared daemon), allowing secure remote access without opening local network ports.
⚬ CORS Configured: Strictly allows cross-origin requests from authorized production domains.

==================================================================
Prerequisites
==================================================================
⚬ Python 3.10+
⚬ Ollama server running locally
⚬ A Cloudflare Zero Trust account (for remote tunneling)

==================================================================
Installation & Setup
==================================================================
1. Clone the repository and set up the environment:

git clone https://github.com/yourusername/aalo.git
cd aalo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Configure Environment Variables:
   Create a .env file in the root directory. This file is ignored by git to protect the AI's core instructions and API keys. Use the provided .env.example as a template.

==================================================================
Running the Application
==================================================================
1. Start the Local LLM
Ensure your Ollama service is running in the background.

2. Start the FastAPI Gateway
Run the Uvicorn server on your local machine:

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
or
./run_server.sh

3. Initialize the Public Tunnel (Production)
If configured as a macOS/Linux background daemon, the Cloudflare tunnel will start automatically on boot. To install or run it manually using your Zero Trust token:

sudo cloudflared service install <YOUR_SECRET_TOKEN>

The gateway is now accessible securely via your designated Cloudflare subdomain (e.g., [https://api.yourdomain.com]).

==================================================================
Security Notes
==================================================================
⚬ Prompt Protection: The endpoint explicitly strips incoming messages tagged with the "system" role to prevent client-side prompt injection.
⚬ Environment Secrets: Never commit the .env file. Rely on pydantic_settings for safe credential loading.