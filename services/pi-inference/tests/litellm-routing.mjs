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
const receivedBetas = [];
const limited = { busy: 0, quota: 0 };
const quotaReset = Math.floor(Date.now() / 1000) + 3600;
const stub = createServer(async (req, res) => {
  const parts = [];
  for await (const p of req) parts.push(p);
  const body = JSON.parse(Buffer.concat(parts));
  received.push(body);
  receivedBetas.push(req.headers["anthropic-beta"] ?? "");
  const asked = JSON.stringify(body.messages ?? []);
  const limit = asked.includes("busy now")
    ? "busy"
    : asked.includes("over quota")
      ? "quota"
      : undefined;
  if (limit) {
    limited[limit] += 1;
    res.writeHead(429, {
      "content-type": "application/json",
      ...(limit === "busy"
        ? { "retry-after": "1" }
        : {
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-reset": String(quotaReset),
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
          }),
    });
    res.end(
      JSON.stringify({
        type: "error",
        error: { type: "rate_limit_error", message: "Rate limited" },
      }),
    );
    return;
  }
  const toolName = body.tools?.find(
    (tool) => !tool.type || tool.type === "custom",
  )?.name;
  res.writeHead(200, { "content-type": "text/event-stream" });
  assert.equal(
    new URL(req.url, "http://stub.invalid").pathname,
    "/v1/messages",
  );
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
        alias: "claude-haiku-4-5",
        provider: "anthropic",
        model: "claude-haiku-4-5",
        baseUrl: stubUrl,
        metadata: metadata("anthropic-messages"),
      },
      {
        alias: "claude-opus-5-5",
        provider: "anthropic",
        model: "claude-opus-5-5",
        baseUrl: stubUrl,
      },
      {
        alias: "claude-sonnet-5-5",
        provider: "anthropic",
        model: "claude-sonnet-5-5",
        baseUrl: stubUrl,
      },
    ],
  },
  {
    authContext: {
      env: async (name) =>
        name === "ANTHROPIC_API_KEY" ? "sk-ant-oat-fixture" : undefined,
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
    r=await litellm.acompletion(model='anthropic/claude-haiku-4-5',api_base=base,api_key=key,max_tokens=128,messages=[{'role':'user','content':'hello'}],stream=stream,**({'stream_options':{'include_usage':True}} if stream else {}))
    if stream:
      chunks=[x async for x in r]
      assert ''.join(x.choices[0].delta.content or '' for x in chunks if x.choices)=='local routing works'
    else:
      assert r.choices[0].message.content=='local routing works'
      assert r.usage.total_tokens==13
    print('LiteLLM Chat client via native Messages stream='+str(stream)+' PASS')
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
    r=await litellm.acompletion(model='anthropic/claude-haiku-4-5',api_base=base,api_key=key,max_tokens=128,messages=[{'role':'system','content':'pi itself and pi packages'},*messages],tools=[{'type':'function','function':{'name':'lookup_weather','parameters':schema}}],stream=stream)
    if stream:
      chunks=[x async for x in r]
      calls=[c for x in chunks if x.choices for c in (x.choices[0].delta.tool_calls or [])]
      assert ''.join(c.function.name or '' for c in calls)=='lookup_weather',calls
      assert json.loads(''.join(c.function.arguments or '' for c in calls))=={'city':'Vienna'},calls
    else:
      call=r.choices[0].message.tool_calls[0]
      assert call.function.name=='lookup_weather',r
      assert json.loads(call.function.arguments)=={'city':'Vienna'},r
    print('LiteLLM OAuth tools Chat client via native Messages stream='+str(stream)+' PASS')
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
  opus_messages=[{'role':'user','content':'hello'},{'role':'system','content':[],'output_config':{'effort':'xhigh'}}]
  for stream in (False,True):
    r=await acreate(model='anthropic/claude-opus-5-5',api_base=base,api_key=key,max_tokens=64000,messages=opus_messages,thinking={'type':'adaptive','display':'summarized','block_binding':{'prefix_mismatch_behavior':'error'}},output_config={'effort':'xhigh'},extra_headers={'anthropic-beta':'mid-conversation-output-config-2026-07-01,thinking-binding-controls-2026-08-01'},stream=stream)
    if stream:
      chunks=[x async for x in r]
      wire=''.join(x.decode() if isinstance(x,bytes) else x if isinstance(x,str) else json.dumps(x) for x in chunks)
      assert 'local routing works' in wire and 'message_stop' in wire,wire
    else:
      assert r['content'][0]['text']=='local routing works',r
    print('LiteLLM Opus 5.5 Messages stream='+str(stream)+' PASS')
  sonnet_messages=[{'role':'user','content':[{'type':'document','source':{'type':'text','media_type':'text/plain','data':'forecast'},'citations':{'enabled':True}},{'type':'text','text':'hello'}]}]
  for stream in (False,True):
    r=await acreate(model='anthropic/claude-sonnet-5-5',api_base=base,api_key=key,max_tokens=1024,messages=sonnet_messages,stop_sequences=['END'],tools=[{'type':'web_search_20250305','name':'web_search','max_uses':1}],stream=stream)
    if stream:
      chunks=[x async for x in r]
      wire=''.join(x.decode() if isinstance(x,bytes) else x if isinstance(x,str) else json.dumps(x) for x in chunks)
      assert 'local routing works' in wire and 'message_stop' in wire,wire
    else:
      assert r['content'][0]['text']=='local routing works',r
    print('LiteLLM Sonnet 5.5 Messages with a document, hosted tool and stop sequence stream='+str(stream)+' PASS')
  for tool_choice in ('required',{'type':'function','function':{'name':'lookup_weather'}}):
    try:
      await litellm.acompletion(model='anthropic/claude-sonnet-5-5',api_base=base,api_key=key,max_tokens=128,messages=messages,tools=[{'type':'function','function':{'name':'lookup_weather','parameters':schema}}],tool_choice=tool_choice,drop_params=True)
    except litellm.BadRequestError as error:
      assert 'forced tool use' in str(error),error
    else:
      raise AssertionError('forced tool choice reached the backend')
  print('LiteLLM Chat forced tool choice on Sonnet 5.5 fails closed with drop_params PASS')
  from litellm.router_utils.subscription_exhaustion import subscription_exhausted_until
  reset=int(os.environ['CHECK_QUOTA_RESET'])
  for text,expected in (('busy now',None),('over quota',reset)):
    for stream in (False,True):
      for surface in ('chat','messages'):
        request=dict(model='anthropic/claude-haiku-4-5',api_base=base,api_key=key,max_tokens=16,messages=[{'role':'user','content':text}],stream=stream)
        try:
          r=await (litellm.acompletion(**request) if surface=='chat' else acreate(**request))
          if stream:
            _=[x async for x in r]
        except litellm.RateLimitError as error:
          assert subscription_exhausted_until(error)==expected,(surface,stream,error)
        else:
          raise AssertionError(f'{text} answered on {surface}')
    print(f'LiteLLM reads {text!r} as '+('load' if expected is None else 'an account out of quota')+' on Chat and Messages PASS')
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
        CHECK_QUOTA_RESET: String(quotaReset),
        LITELLM_LOCAL_MODEL_COST_MAP: "True",
      },
      timeout: 60000,
      maxBuffer: 1024 * 1024,
    },
  );
  console.log(stdout);
  if (stderr) console.error(stderr);
  assert.deepEqual(limited, { busy: 4, quota: 1 });
  const opus = received
    .map((payload, index) => ({ payload, betas: receivedBetas[index] }))
    .filter(({ payload }) => payload.model === "claude-opus-5-5");
  assert.equal(opus.length, 2);
  for (const { payload, betas } of opus) {
    assert.deepEqual(payload.messages.at(-1), {
      role: "system",
      content: [],
      output_config: { effort: "xhigh" },
    });
    assert.deepEqual(payload.thinking, {
      type: "adaptive",
      display: "summarized",
      block_binding: { prefix_mismatch_behavior: "error" },
    });
    assert.deepEqual(payload.output_config, { effort: "xhigh" });
    assert.equal(payload.max_tokens, 64000);
    for (const beta of [
      "mid-conversation-output-config-2026-07-01",
      "thinking-binding-controls-2026-08-01",
      "oauth-2025-04-20",
    ])
      assert(betas.split(",").includes(beta), betas);
  }
  const sonnet = received.filter(
    (payload) => payload.model === "claude-sonnet-5-5",
  );
  assert.equal(sonnet.length, 2);
  for (const payload of sonnet) {
    assert.deepEqual(payload.stop_sequences, ["END"]);
    assert.deepEqual(payload.tools, [
      { type: "web_search_20250305", name: "web_search", max_uses: 1 },
    ]);
    assert.deepEqual(payload.messages[0].content[0], {
      type: "document",
      source: { type: "text", media_type: "text/plain", data: "forecast" },
      citations: { enabled: true },
    });
  }
  const toolRequests = received.filter(
    (payload) => payload.model === "claude-haiku-4-5" && payload.tools?.length,
  );
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
