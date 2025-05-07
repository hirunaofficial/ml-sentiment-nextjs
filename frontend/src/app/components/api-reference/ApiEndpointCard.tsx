import React from 'react';

interface ApiEndpointCardProps {
  method: string;
  endpoint: string;
  title: string;
  requestExample: string;
  responseExample: string;
  colSpan?: string;
}

const ApiEndpointCard: React.FC<ApiEndpointCardProps> = ({
  method,
  endpoint,
  title,
  requestExample,
  responseExample,
  colSpan = "",
}) => {
  return (
    <div className={`bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-black/[.05] dark:border-white/[.08] ${colSpan}`}>
      <div className="flex flex-col md:flex-row md:items-center gap-2 mb-2">
        <span className="px-2 py-1 bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-400 rounded text-xs font-medium">
          {method}
        </span>
        <code className="bg-black/[.05] dark:bg-white/[.08] px-2 py-1 rounded font-[family-name:var(--font-geist-mono)] text-xs">
          {endpoint}
        </code>
      </div>
      <h3 className="text-sm font-medium mb-2 mt-3">
        {title}
      </h3>
      <div className="bg-white dark:bg-black/40 p-3 rounded-lg text-xs font-[family-name:var(--font-geist-mono)] overflow-x-auto border border-black/[.05] dark:border-white/[.08]">
        <pre className="whitespace-pre-wrap">{`Request:
${requestExample}

Response:
${responseExample}`}</pre>
      </div>
    </div>
  );
};

export default ApiEndpointCard;