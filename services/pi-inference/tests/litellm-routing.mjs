import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { pathToFileURL, fileURLToPath } from "node:url";
import { resolve } from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url));
const forkRoot = resolve(root, "../..");
const { loadRuntime } = await import(pathToFileURL(`${root}/dist/runtime.js`));
const { createInferenceServer } = await import(
  pathToFileURL(`${root}/dist/server.js`)
);
const key = "local-test-internal-key-123456789012345";
const listen = async (server) => {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return `http://127.0.0.1:${server.address().port}`;
};
const received = [];
const stub = createServer(async (req, res) => {
  const parts = [];
  for await (const p of req) parts.push(p);
  const body = JSON.parse(Buffer.concat(parts));
  received.push(body);
  const toolName = body.tools?.[0]?.name;
  res.writeHead(200, { "content-type": "text/event-stream" });
  if (req.url === "/v1/messages") {
    for (const ev of [
      {
        type: "message_start",
        message: {
          id: "msg_stub",
          type: "message",
          role: "assistant",
          model: body.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 10, output_tokens: 0 },
        },
      },
      {
        type: "content_block_start",
        index: 0,
        content_block: toolName
          ? { type: "tool_use", id: "tool_fixture", name: toolName, input: {} }
          : { type: "text", text: "" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: toolName
          ? { type: "input_json_delta", partial_json: '{"city":"Vienna"}' }
          : { type: "text_delta", text: "local routing works" },
      },
      { type: "content_block_stop", index: 0 },
      {
        type: "message_delta",
        delta: {
          stop_reason: toolName ? "tool_use" : "end_turn",
          stop_sequence: null,
        },
        usage: { output_tokens: 3 },
      },
      { type: "message_stop" },
    ])
      res.write(`event: ${ev.type}\ndata: ${JSON.stringify(ev)}\n\n`);
  } else {
    for (const ev of [
      {
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "local routing works" },
            finish_reason: null,
          },
        ],
      },
      {
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
      },
    ])
      res.write(
        `data: ${JSON.stringify({ id: "chat_stub", object: "chat.completion.chunk", model: body.model, created: 0, ...ev })}\n\n`,
      );
    res.write("data: [DONE]\n\n");
  }
  res.end();
});
const stubUrl = await listen(stub);
const metadata = (api) => ({
  api,
  contextWindow: 4096,
  maxTokens: 1024,
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
});
const runtime = loadRuntime(
  {
    models: [
      {
        alias: "chat",
        provider: "stub-chat",
        model: "stub",
        baseUrl: stubUrl + "/v1",
        metadata: metadata("openai-completions"),
      },
      {
        alias: "claude-haiku-4-5",
        provider: "anthropic",
        model: "claude-haiku-4-5",
        baseUrl: stubUrl,
        metadata: metadata("anthropic-messages"),
      },
    ],
  },
  {
    authContext: {
      env: async (name) =>
        name === "ANTHROPIC_API_KEY"
          ? "sk-ant-oat-fixture"
          : name === "STUB_CHAT_API_KEY"
            ? "upstream-fixture-key"
            : undefined,
      fileExists: async () => false,
    },
  },
);
if (!runtime.ok) throw Error(JSON.stringify(runtime));
const backend = createInferenceServer({
  apiKey: key,
  runtime: runtime.value,
  log: (r) => console.log(JSON.stringify(r)),
});
const base = await listen(backend.server);
const script = `import asyncio,json,os
import litellm
litellm.telemetry = False
from litellm.anthropic_interface import acreate
async def main():
  base=os.environ['CHECK_BASE']; key=os.environ['CHECK_KEY']
  for stream in (False,True):
    r=await litellm.acompletion(model='litellm_proxy/chat',api_base=base+'/v1',api_key=key,messages=[{'role':'user','content':'hello'}],stream=stream,**({'stream_options':{'include_usage':True}} if stream else {}))
    if stream:
      chunks=[x async for x in r]
      assert ''.join(x.choices[0].delta.content or '' for x in chunks if x.choices)=='local routing works'
    else:
      assert r.choices[0].message.content=='local routing works'
      assert r.usage.total_tokens==13
    print('LiteLLM chat stream='+str(stream)+' PASS')
  for stream in (False,True):
    r=await acreate(model='anthropic/claude-haiku-4-5',api_base=base,api_key=key,max_tokens=128,messages=[{'role':'user','content':'hello'}],stream=stream)
    if stream:
      chunks=[x async for x in r]
      wire=''.join(x.decode() if isinstance(x,bytes) else x if isinstance(x,str) else json.dumps(x) for x in chunks)
      assert 'local routing works' in wire,wire
      assert 'message_stop' in wire,wire
      assert 'output_tokens' in wire,wire
    else:
      assert r['content'][0]['text']=='local routing works',r
      assert r['usage']['input_tokens']==10,r
    print('LiteLLM Messages stream='+str(stream)+' PASS')
  schema={'type':'object','properties':{'city':{'type':'string'}}}
  messages=[{'role':'user','content':'pi itself is user text'}]
  for stream in (False,True):
    r=await litellm.acompletion(model='litellm_proxy/claude-haiku-4-5',api_base=base+'/v1',api_key=key,messages=[{'role':'system','content':'pi itself and pi packages'},*messages],tools=[{'type':'function','function':{'name':'lookup_weather','parameters':schema}}],stream=stream)
    if stream:
      chunks=[x async for x in r]
      calls=[c for x in chunks if x.choices for c in (x.choices[0].delta.tool_calls or [])]
      assert ''.join(c.function.name or '' for c in calls)=='lookup_weather',calls
      assert json.loads(''.join(c.function.arguments or '' for c in calls))=={'city':'Vienna'},calls
    else:
      call=r.choices[0].message.tool_calls[0]
      assert call.function.name=='lookup_weather',r
      assert json.loads(call.function.arguments)=={'city':'Vienna'},r
    print('LiteLLM OAuth tools Chat stream='+str(stream)+' PASS')
    r=await acreate(model='anthropic/claude-haiku-4-5',api_base=base,api_key=key,max_tokens=128,messages=messages,system='pi itself and pi packages',tools=[{'name':'lookup_weather','input_schema':schema}],tool_choice={'type':'tool','name':'lookup_weather'},stream=stream)
    if stream:
      chunks=[x async for x in r]
      wire=''.join(x.decode() if isinstance(x,bytes) else x if isinstance(x,str) else json.dumps(x) for x in chunks)
      assert 'lookup_weather' in wire and 'mcp__pi__' not in wire,wire
      assert 'message_stop' in wire and 'output_tokens' in wire,wire
    else:
      assert r['content'][0]['name']=='lookup_weather',r
      assert r['content'][0]['input']=={'city':'Vienna'},r
      assert r['usage']['input_tokens']==10,r
    print('LiteLLM OAuth tools Messages stream='+str(stream)+' PASS')
  await asyncio.sleep(0.1)
asyncio.run(main())`;
try {
  const { stdout, stderr } = await promisify(execFile)(
    "uv",
    ["run", "--project", forkRoot, "--no-sync", "python", "-c", script],
    {
      env: {
        ...process.env,
        PYTHONPATH: forkRoot,
        CHECK_BASE: base,
        CHECK_KEY: key,
        LITELLM_LOCAL_MODEL_COST_MAP: "True",
      },
      timeout: 60000,
      maxBuffer: 1024 * 1024,
    },
  );
  console.log(stdout);
  if (stderr) console.error(stderr);
  const toolRequests = received.filter((payload) => payload.tools?.length);
  assert.equal(toolRequests.length, 4);
  for (const payload of toolRequests) {
    assert.equal(payload.tools[0].name, "mcp__pi__lookup_weather");
    assert.equal(payload.system.at(-1).text, "the cli itself and cli packages");
    assert(JSON.stringify(payload.messages).includes("pi itself is user text"));
    if (payload.tool_choice?.type === "tool")
      assert.equal(payload.tool_choice.name, "mcp__pi__lookup_weather");
  }
} finally {
  backend.abortAll();
  backend.server.closeAllConnections();
  stub.closeAllConnections();
  await Promise.all([
    new Promise((r) => backend.server.close(r)),
    new Promise((r) => stub.close(r)),
  ]);
}
