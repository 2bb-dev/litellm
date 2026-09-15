import { createServer } from "node:http";
import { pathToFileURL } from "node:url";

const { handleTerminalReceipts, closeTerminalReceiptRuntime } = await import(
  pathToFileURL(`${process.env.OPENORANGE_TEST_PLATFORM_CHECKOUT}/src/server/terminal-receipts/service.ts`).href
);
const server = createServer(async (incoming, outgoing) => {
  const chunks = [];
  for await (const chunk of incoming) chunks.push(chunk);
  const body = Buffer.concat(chunks);
  const request = new Request(`http://127.0.0.1${incoming.url}`, {
    method: incoming.method,
    headers: incoming.headers,
    ...(body.length ? { body } : {}),
  });
  const response = await handleTerminalReceipts(request);
  outgoing.writeHead(response.status, Object.fromEntries(response.headers));
  outgoing.end(Buffer.from(await response.arrayBuffer()));
});
server.listen(0, "127.0.0.1", () => process.stdout.write(JSON.stringify({ port: server.address().port }) + "\n"));
process.on("SIGTERM", () => server.close(() => { closeTerminalReceiptRuntime(); process.exit(0); }));
