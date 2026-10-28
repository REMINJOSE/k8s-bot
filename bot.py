import os
import json
import asyncio
import subprocess
import discord
from discord import app_commands
import requests
from aiohttp import web

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DISCORD_ALERT_CHANNEL_ID = int(os.getenv("DISCORD_ALERT_CHANNEL_ID", "0"))
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8080"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q5_K_M")

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def run_command(cmd: list[str], max_chars: int = 3500) -> str:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=90)
        output = res.stdout if res.stdout.strip() else res.stderr
        if not output.strip():
            return "Command succeeded with no output (empty result)."
        if len(output) > max_chars:
            return output[:max_chars] + "\n...[Output truncated to fit context window]..."
        return output
    except Exception as e:
        return f"Execution error: {str(e)}"

# ==========================================
# 1. READ-ONLY TOOL IMPLEMENTATIONS
# ==========================================

def get_k8s_resources(resource_type: str, namespace: str = "all") -> str:
    allowed = {"pods", "pod", "svc", "services", "deployments", "deploy", "nodes", "pvc", "ingress", "statefulsets", "sts", "daemonsets", "ds", "namespaces", "ns"}
    r_clean = resource_type.strip().lower().split()[0]
    if r_clean not in allowed:
        return f"Unsupported resource '{r_clean}'."
    cmd = ["kubectl", "get", r_clean, "-o", "wide"]
    if r_clean not in {"nodes", "namespaces", "ns"}:
        if namespace.lower() in ["all", "", "*", "-a"]:
            cmd.append("-A")
        else:
            cmd.extend(["-n", namespace.strip().split()[0]])
    return run_command(cmd)

def describe_k8s_resource(resource_type: str, name: str, namespace: str = "default") -> str:
    return run_command(["kubectl", "describe", resource_type.strip(), name.strip(), "-n", namespace.strip()])

def get_pod_logs(pod_name: str, namespace: str = "default", lines: int = 50, previous: bool = False) -> str:
    cmd = ["kubectl", "logs", pod_name.strip(), "-n", namespace.strip(), f"--tail={min(int(lines), 100)}"]
    if previous:
        cmd.append("--previous")
    return run_command(cmd)

def get_k8s_events(namespace: str = "all") -> str:
    cmd = ["kubectl", "get", "events", "--sort-by=.lastTimestamp"]
    if namespace.lower() in ["all", "", "*"]:
        cmd.append("-A")
    else:
        cmd.extend(["-n", namespace.strip().split()[0]])
    return run_command(cmd)

def run_k8sgpt_scan(namespace: str = "all") -> str:
    cmd = ["k8sgpt", "analyze", "--explain", "--backend", "localai"]
    if namespace.lower() not in ["all", "", "*"]:
        cmd.extend(["--namespace", namespace.strip().split()[0]])
    return run_command(cmd)

def get_docker_containers(all_containers: bool = True) -> str:
    cmd = ["docker", "ps"]
    if all_containers:
        cmd.append("-a")
    cmd.extend(["--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.RunningFor}}"])
    return run_command(cmd)

def get_docker_logs(container_name: str, lines: int = 50) -> str:
    return run_command(["docker", "logs", f"--tail={min(int(lines), 100)}", container_name.strip()])

def get_docker_stats() -> str:
    return run_command(["docker", "stats", "--no-stream", "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"])

def check_docker_db_redis(postgres_container: str = "immich_postgres", redis_container: str = "immich_redis") -> str:
    pg_status = run_command(["docker", "exec", postgres_container.strip(), "pg_isready"], 500)
    redis_status = run_command(["docker", "exec", redis_container.strip(), "redis-cli", "ping"], 500)
    return f"=== Postgres ===\n{pg_status}\n=== Redis ===\n{redis_status}"

def get_lxc_and_gpu_stats() -> str:
    uptime = run_command(["uptime"], 500)
    mem = run_command(["free", "-h"], 500)
    disk = run_command(["df", "-h", "/"], 500)
    gpu = run_command(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv"], 500)
    return f"=== Uptime ===\n{uptime}\n=== Memory ===\n{mem}\n=== Disk ===\n{disk}\n=== GPU Status ===\n{gpu}"

# ==========================================
# 2. TOOL SCHEMAS & ROUTER
# ==========================================

K8S_TOOLS = [
    {"type": "function", "function": {"name": "get_k8s_resources", "description": "List live Kubernetes resources.", "parameters": {"type": "object", "properties": {"resource_type": {"type": "string"}, "namespace": {"type": "string"}}, "required": ["resource_type"]}}},
    {"type": "function", "function": {"name": "describe_k8s_resource", "description": "Run kubectl describe.", "parameters": {"type": "object", "properties": {"resource_type": {"type": "string"}, "name": {"type": "string"}, "namespace": {"type": "string"}}, "required": ["resource_type", "name", "namespace"]}}},
    {"type": "function", "function": {"name": "get_pod_logs", "description": "Fetch pod logs.", "parameters": {"type": "object", "properties": {"pod_name": {"type": "string"}, "namespace": {"type": "string"}, "lines": {"type": "integer"}, "previous": {"type": "boolean"}}, "required": ["pod_name", "namespace"]}}},
    {"type": "function", "function": {"name": "get_k8s_events", "description": "Get cluster events.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}, "required": ["namespace"]}}},
    {"type": "function", "function": {"name": "run_k8sgpt_scan", "description": "Run K8sGPT analyzers.", "parameters": {"type": "object", "properties": {"namespace": {"type": "string"}}, "required": ["namespace"]}}},
    {"type": "function", "function": {"name": "propose_k8s_restart", "description": "Stage a Kubernetes restart.", "parameters": {"type": "object", "properties": {"resource_type": {"type": "string"}, "name": {"type": "string"}, "namespace": {"type": "string"}, "reason": {"type": "string"}}, "required": ["resource_type", "name", "namespace", "reason"]}}}
]

DOCKER_TOOLS = [
    {"type": "function", "function": {"name": "get_docker_containers", "description": "List local Docker containers.", "parameters": {"type": "object", "properties": {"all_containers": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "get_docker_logs", "description": "Fetch container logs.", "parameters": {"type": "object", "properties": {"container_name": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["container_name"]}}},
    {"type": "function", "function": {"name": "get_docker_stats", "description": "Get container resource stats.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "check_docker_db_redis", "description": "Check Postgres and Redis health.", "parameters": {"type": "object", "properties": {"postgres_container": {"type": "string"}, "redis_container": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "propose_docker_restart", "description": "Stage a Docker container restart.", "parameters": {"type": "object", "properties": {"container_name": {"type": "string"}, "reason": {"type": "string"}}, "required": ["container_name", "reason"]}}}
]

SYSTEM_TOOLS = [{"type": "function", "function": {"name": "get_lxc_and_gpu_stats", "description": "Check host and GPU stats.", "parameters": {"type": "object", "properties": {}}}}]

AVAILABLE_FUNCTIONS = {
    "get_k8s_resources": get_k8s_resources, "describe_k8s_resource": describe_k8s_resource,
    "get_pod_logs": get_pod_logs, "get_k8s_events": get_k8s_events, "run_k8sgpt_scan": run_k8sgpt_scan,
    "get_docker_containers": get_docker_containers, "get_docker_logs": get_docker_logs,
    "get_docker_stats": get_docker_stats, "check_docker_db_redis": check_docker_db_redis,
    "get_lxc_and_gpu_stats": get_lxc_and_gpu_stats,
}

def select_tools(prompt: str) -> list[dict]:
    p_lower = prompt.lower()
    has_docker = any(k in p_lower for k in ["docker", "container", "immich", "postgres", "redis", "ml"])
    has_k8s = any(k in p_lower for k in ["k8s", "k3s", "pod", "namespace", "svc", "deploy", "node", "quiz"])
    
    selected = list(SYSTEM_TOOLS)
    if has_docker and not has_k8s:
        selected.extend(DOCKER_TOOLS)
    elif has_k8s and not has_docker:
        selected.extend(K8S_TOOLS)
    else:
        selected.extend(K8S_TOOLS)
        selected.extend(DOCKER_TOOLS)
    return selected

# ==========================================
# 3. INTERACTIVE APPROVAL UI VIEW
# ==========================================

class RemediationApprovalView(discord.ui.View):
    def __init__(self, actions: list[dict], requester_id: int = None):
        super().__init__(timeout=120)
        self.actions = actions
        self.requester_id = requester_id
        self.message = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if self.requester_id and interaction.user.id != self.requester_id:
            await interaction.response.send_message("❌ Only the authorized user can approve this action.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Approve & Restart", style=discord.ButtonStyle.success, emoji="✅")
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)

        results = []
        for act in self.actions:
            output = run_command(act["cmd"], max_chars=1000)
            results.append(f"**Executed:** `{' '.join(act['cmd'])}`\n```\n{output.strip()}\n```")

        await interaction.followup.send(f"✅ **Remediation Executed by {interaction.user.mention}:**\n" + "\n".join(results))
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="✖️")
    async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)
        await interaction.followup.send("🛑 **Action cancelled.**")
        self.stop()

# ==========================================
# 4. AGENT ENGINE
# ==========================================

def stage_remediation(fn_name: str, fn_args: dict, pending: list[dict]) -> str:
    if fn_name == "propose_docker_restart":
        container = fn_args.get("container_name", "").strip().split()[0]
        cmd = ["docker", "restart", container]
        pending.append({"title": f"Restart Container `{container}`", "cmd": cmd, "reason": fn_args.get("reason", "")})
        return f"STAGED: `docker restart {container}` queued for approval."
    elif fn_name == "propose_k8s_restart":
        r_type = fn_args.get("resource_type", "deployment").strip().lower()
        name = fn_args.get("name", "").strip()
        ns = fn_args.get("namespace", "default").strip()
        cmd = ["kubectl", "rollout", "restart", f"{r_type}/{name}", "-n", ns] if r_type != "pod" else ["kubectl", "delete", "pod", name, "-n", ns]
        pending.append({"title": f"Restart K8s `{r_type}/{name}` (ns: `{ns}`)", "cmd": cmd, "reason": fn_args.get("reason", "")})
        return f"STAGED: `{' '.join(cmd)}` queued for approval."
    return "Unknown."

def run_devops_agent(prompt: str) -> tuple[str, list[str], list[dict]]:
    active_tools = select_tools(prompt)
    system_prompt = "You are an expert DevOps SRE Agent managing k3s, Docker containers, and an RTX 2060 SUPER GPU. Inspect logs and propose restarts when services fail."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    tools_used, pending_actions = [], []

    for _ in range(4):
        payload = {"model": MODEL_NAME, "messages": messages, "tools": active_tools, "stream": False, "options": {"temperature": 0.1, "num_ctx": 8192}}
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        data = resp.json()
        msg = data.get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "No analysis returned."), tools_used, pending_actions

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"].get("arguments", "{}")) if isinstance(tc["function"].get("arguments"), str) else tc["function"].get("arguments", {})
            tools_used.append(fn_name)

            if fn_name in {"propose_docker_restart", "propose_k8s_restart"}:
                output = stage_remediation(fn_name, fn_args, pending_actions)
            else:
                func = AVAILABLE_FUNCTIONS.get(fn_name)
                output = func(**fn_args) if func else f"Error: Unknown tool '{fn_name}'"

            messages.append({"role": "tool", "content": str(output)})

    payload = {"model": MODEL_NAME, "messages": messages, "stream": False, "options": {"temperature": 0.1, "num_ctx": 8192}}
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    return resp.json().get("message", {}).get("content", "Timed out."), tools_used, pending_actions

# ==========================================
# 5. WEBHOOK & DISCORD LIFECYCLE
# ==========================================

async def handle_alert(request: web.Request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    summary = data.get("message", "Alert Triggered")
    prompt = f"🚨 PROACTIVE ALERT RECEIVED: {summary}\nInspect the system and propose a fix."

    reply, tools_used, pending_actions = await asyncio.to_thread(run_devops_agent, prompt)

    channel = client.get_channel(DISCORD_ALERT_CHANNEL_ID)
    if channel:
        header = f"🚨 **Proactive Alert Diagnosis**\n**Trigger:** `{summary}`\n"
        if tools_used:
            header += f"**Tools:** " + " → ".join([f"`{t}`" for t in tools_used]) + "\n\n"
        view = RemediationApprovalView(pending_actions) if pending_actions else None
        msg = await channel.send(header + reply, view=view)
        if view:
            view.message = msg

    return web.json_response({"status": "success"})

async def start_webhook():
    app = web.Application()
    app.router.add_post("/webhook", handle_alert)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", WEBHOOK_PORT)
    await site.start()
    print(f"🚀 Webhook listening on port {WEBHOOK_PORT}")

@tree.command(name="ask", description="Ask the DevOps Agent a question")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    reply, tools_used, pending_actions = await asyncio.to_thread(run_devops_agent, prompt)
    header = ("**Tools:** " + " → ".join([f"`{t}`" for t in tools_used]) + "\n\n") if tools_used else ""
    view = RemediationApprovalView(pending_actions, interaction.user.id) if pending_actions else None
    await interaction.followup.send(header + reply, view=view)

@client.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {client.user} - Master DevOps Agent ready.")
    asyncio.create_task(start_webhook())

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
