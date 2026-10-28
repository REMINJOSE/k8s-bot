import os
import json
import subprocess
import discord
from discord import app_commands
import requests

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
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

# --- TOOL IMPLEMENTATIONS ---

def get_k8s_resources(resource_type: str, namespace: str = "all") -> str:
    allowed = {"pods", "pod", "svc", "services", "deployments", "deploy", "nodes", "pvc", "ingress", "statefulsets", "sts", "daemonsets", "ds", "namespaces", "ns"}
    r_clean = resource_type.strip().lower().split()[0]
    if r_clean not in allowed:
        return f"Unsupported resource '{r_clean}'. Allowed: {', '.join(sorted(allowed))}"
    
    cmd = ["kubectl", "get", r_clean, "-o", "wide"]
    if r_clean not in {"nodes", "namespaces", "ns"}:
        if namespace.lower() in ["all", "", "*", "-a"]:
            cmd.append("-A")
        else:
            cmd.extend(["-n", namespace.strip().split()[0]])
    return run_command(cmd)

def describe_k8s_resource(resource_type: str, name: str, namespace: str = "default") -> str:
    r_clean = resource_type.strip().lower().split()[0]
    n_clean = name.strip().split()[0]
    ns_clean = namespace.strip().split()[0]
    cmd = ["kubectl", "describe", r_clean, n_clean, "-n", ns_clean]
    return run_command(cmd)

def get_pod_logs(pod_name: str, namespace: str = "default", lines: int = 50, previous: bool = False) -> str:
    p_clean = pod_name.strip().split()[0]
    ns_clean = namespace.strip().split()[0]
    cmd = ["kubectl", "logs", p_clean, "-n", ns_clean, f"--tail={min(int(lines), 100)}"]
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

def get_lxc_system_stats() -> str:
    uptime = run_command(["uptime"], 500)
    mem = run_command(["free", "-h"], 500)
    disk = run_command(["df", "-h", "/"], 500)
    return f"=== Uptime / Load ===\n{uptime}\n=== Memory ===\n{mem}\n=== Disk ===\n{disk}"

# --- OLLAMA TOOL SCHEMAS ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_k8s_resources",
            "description": "List live Kubernetes resources (pods, svc, deployments, nodes, pvc, ingress, statefulsets, namespaces). Always call this when asked what is running in a namespace or across the cluster.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_type": {"type": "string", "description": "e.g., pods, svc, deployments, nodes, pvc, ingress"},
                    "namespace": {"type": "string", "description": "Target namespace name, or 'all' for all namespaces"}
                },
                "required": ["resource_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_k8s_resource",
            "description": "Run 'kubectl describe' on a specific pod, node, deployment, pvc, or service to inspect status conditions, image pull errors, or mount issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_type": {"type": "string", "description": "Resource type (pod, deployment, svc, pvc, node)"},
                    "name": {"type": "string", "description": "Exact name of the resource"},
                    "namespace": {"type": "string", "description": "Namespace of the resource"}
                },
                "required": ["resource_type", "name", "namespace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_logs",
            "description": "Fetch recent stdout/stderr application logs from a specific pod.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod_name": {"type": "string", "description": "Exact name of the pod"},
                    "namespace": {"type": "string", "description": "Namespace of the pod"},
                    "lines": {"type": "integer", "description": "Number of log lines to fetch (default 50)"},
                    "previous": {"type": "boolean", "description": "Set to true if the pod is CrashLooping and you want the crashed container's logs"}
                },
                "required": ["pod_name", "namespace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_k8s_events",
            "description": "Get recent Kubernetes cluster events (warnings, scheduling failures, OOMKills, volume issues).",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace to filter events, or 'all'"}
                },
                "required": ["namespace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_k8sgpt_scan",
            "description": "Run automated K8sGPT diagnostic analyzers across the cluster or a specific namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Target namespace or 'all'"}
                },
                "required": ["namespace"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_lxc_system_stats",
            "description": "Check CPU load, RAM usage, and disk space on the local LXC container host.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

AVAILABLE_FUNCTIONS = {
    "get_k8s_resources": get_k8s_resources,
    "describe_k8s_resource": describe_k8s_resource,
    "get_pod_logs": get_pod_logs,
    "get_k8s_events": get_k8s_events,
    "run_k8sgpt_scan": run_k8sgpt_scan,
    "get_lxc_system_stats": get_lxc_system_stats,
}

# --- AGENT REASONING LOOP ---

def run_devops_agent(user_prompt: str) -> tuple[str, list[str]]:
    system_prompt = (
        "You are an expert DevOps & SRE Troubleshooting Agent with live read-only access to a k3s Kubernetes cluster and LXC environment.\n"
        "CRITICAL RULES:\n"
        "1. NEVER give generic tutorial explanations on how to run kubectl commands. ALWAYS call your tools first to inspect the live system.\n"
        "2. If asked about pods, services, or errors in a namespace, call 'get_k8s_resources' first.\n"
        "3. If a pod is failing (CrashLoopBackOff, Error, Pending), automatically follow up by calling 'get_pod_logs' or 'describe_k8s_resource' to find the root cause before answering.\n"
        "4. Base your answer strictly on the real tool outputs returned. Be concise and list exact pod names, statuses, and actionable fixes."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    executed_tools_log = []

    # Allow up to 4 sequential tool-calling steps for multi-hop troubleshooting
    for _ in range(4):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for reliable tool calling
                "num_ctx": 8192      # Fits comfortably in 8GB VRAM on RTX 2060 SUPER
            }
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # Model finished gathering context and produced the final answer
            return msg.get("content", "No response generated."), executed_tools_log

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"].get("arguments", {})
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}

            executed_tools_log.append(f"`{fn_name}({json.dumps(fn_args)})`")

            func = AVAILABLE_FUNCTIONS.get(fn_name)
            if func:
                try:
                    tool_output = func(**fn_args)
                except Exception as e:
                    tool_output = f"Tool execution error: {str(e)}"
            else:
                tool_output = f"Error: Unknown tool '{fn_name}'"

            messages.append({
                "role": "tool",
                "content": str(tool_output)
            })

    # If max tool iterations reached, request final synthesis without tools
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192}
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    return resp.json().get("message", {}).get("content", "Analysis timed out."), executed_tools_log

# --- DISCORD COMMANDS ---

async def send_formatted_response(interaction: discord.Interaction, text: str, tools_used: list[str] = None):
    header = ""
    if tools_used:
        header = "**Tools Executed:** " + " → ".join(tools_used) + "\n\n"
    full_text = header + text

    chunk_size = 1900
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        if i == 0:
            await interaction.followup.send(chunk)
        else:
            await interaction.channel.send(chunk)

@tree.command(name="ask", description="Ask the DevOps Agent to inspect and troubleshoot your cluster")
@app_commands.describe(prompt="Your question (e.g., 'What pods are in quiz and are any failing?')")
async def ask(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer(thinking=True)
    try:
        reply, tools_used = run_devops_agent(prompt)
    except Exception as e:
        reply, tools_used = f"Agent error: {str(e)}", []
    await send_formatted_response(interaction, reply, tools_used)

@tree.command(name="analyze", description="Run a direct K8sGPT scan on the cluster")
@app_commands.describe(namespace="Namespace to target (optional, defaults to all)")
async def analyze(interaction: discord.Interaction, namespace: str = "all"):
    await interaction.response.defer(thinking=True)
    output = run_k8sgpt_scan(namespace)
    await send_formatted_response(interaction, f"```\n{output}\n```", [f"`run_k8sgpt_scan(namespace='{namespace}')`"])

@client.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {client.user} - Agent tools & slash commands synced.")

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
