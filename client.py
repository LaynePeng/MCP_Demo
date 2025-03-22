import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

MODEL = config["model"]
API_KEY = config["api_key"]
BASE_URL = config["base_url"]
SYSTEM_PROMPT = config["system_prompt"]
MCP_SERVERS = config['mcp_servers']

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
server_map = {}

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.llm = client
        self.server_map = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools = []

    async def connect_to_server(self, mpc_servers: list):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        for server in config["mcp_servers"]:
            print(f"Setting up server: {server['name']}")
            server_params = StdioServerParameters(
                command=server["command"],
                args=server["args"],
                env=server["env"],
            )
        
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
            await session.initialize()
        
            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            for tool in tools:
                self.server_map[tool.name]=session

        print("\nServer map:", self.server_map)

    async def query_avaiable_tools(self):
        existed_tools = set()
        for session in self.server_map.values():
            response = await session.list_tools()
            for tool in response.tools:
                if tool.name in existed_tools:
                    continue

                existed_tools.add(tool.name)
                self.available_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                })
    
    async def process_query(self, query: str) -> str:
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        if self.available_tools  == []:
            await self.query_avaiable_tools()

        response = await self.llm.chat.completions.create(
            model=MODEL,
            max_tokens=512,
            messages=messages,
            tools=self.available_tools,
            temperature=0, 
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = await self.server_map[tool_name].call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(response.choices[0].message.content, 'text') and response.choices[0].message.text:
                    messages.append({
                      "role": "assistant",
                      "content": result.content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                response = await self.llm.chat.completions.create(
                    model=MODEL,
                    max_tokens=512,
                    messages=messages,
                    temperature=0, 
                )

        final_text.append(response.choices[0].message.content)
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(MCP_SERVERS)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())