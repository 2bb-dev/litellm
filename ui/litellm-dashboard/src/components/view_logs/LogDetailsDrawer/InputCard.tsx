/**
 * InputCard - Displays all input messages with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

import { useState } from 'react';
import { Typography } from 'antd';
import MessageManager from "@/components/molecules/message_manager";
import { ParsedMessage, SenderInfo } from './prettyMessagesTypes';
import { SectionHeader } from './SectionHeader';
import { CollapsibleMessage } from './CollapsibleMessage';
import { HistoryTree } from './HistoryTree';
import { SimpleMessageBlock } from './SimpleMessageBlock';

const { Text } = Typography;

interface InputCardProps {
  messages: ParsedMessage[];
  promptTokens?: number;
  inputCost?: number;
  senderInfo?: SenderInfo;
}

const getPrimaryMessageIndex = (messages: ParsedMessage[]): number => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === 'user' && message.content.trim().length > 0) {
      return index;
    }
  }
  return messages.length > 0 ? messages.length - 1 : -1;
};

export function InputCard({ messages, promptTokens, inputCost, senderInfo }: InputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  // OpenClaw Responses API logs often end with tool output. Keep the latest
  // actual user text visible and leave function/tool history collapsed.
  const systemMessage = messages.find((m) => m.role === 'system');
  const nonSystemMessages = messages.filter((m) => m.role !== 'system');
  const primaryMessageIndex = getPrimaryMessageIndex(nonSystemMessages);
  const primaryMessage = primaryMessageIndex >= 0 ? nonSystemMessages[primaryMessageIndex] : null;
  const historyMessages = nonSystemMessages.filter((_, index) => index !== primaryMessageIndex);

  const handleCopy = () => {
    const content = primaryMessage?.content || '';
    navigator.clipboard.writeText(content);
    MessageManager.success('Input copied');
  };

  return (
    <div
      style={{
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        marginBottom: 8,
        overflow: 'hidden',
      }}
    >
      {/* Datadog-style Header */}
      <SectionHeader
        type="input"
        tokens={promptTokens}
        cost={inputCost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      {/* Content */}
      <div
        style={{
          maxHeight: isCollapsed ? '0px' : '10000px',
          overflow: 'hidden',
          transition: 'max-height 0.3s ease-out, opacity 0.3s ease-out',
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div style={{ padding: '12px 16px' }}>
          {/* System Message - Collapsible with arrow */}
          {systemMessage && (
            <CollapsibleMessage
              label="SYSTEM"
              content={systemMessage.content}
              defaultExpanded={!!(systemMessage.content && systemMessage.content.length < 200)}
            />
          )}

          {/* History - Tree style, collapsed by default */}
          {historyMessages.length > 0 && <HistoryTree messages={historyMessages} />}

          {/* Primary user message - Always visible */}
          {primaryMessage && (
            <>
              {senderInfo && (
                <Text
                  type="secondary"
                  style={{
                    display: 'block',
                    fontSize: 12,
                    marginBottom: 8,
                  }}
                >
                  From {senderInfo.channel ? `${senderInfo.channel}: ` : ''}{senderInfo.label}
                </Text>
              )}
              <SimpleMessageBlock
                label={primaryMessage.role.toUpperCase()}
                content={primaryMessage.content}
                toolCalls={primaryMessage.toolCalls}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
