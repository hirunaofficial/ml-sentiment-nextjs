import React from 'react';
import ApiEndpointCard from './ApiEndpointCard';
import { API_EXAMPLES } from '../../utils/constants';

const ApiReference: React.FC = () => {
  return (
    <div className="w-full bg-white dark:bg-black/40 rounded-xl border border-black/[.08] dark:border-white/[.12] p-6 shadow-sm">
      <h2 className="text-xl font-semibold mb-4 flex items-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mr-2 text-blue-500 dark:text-blue-400"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
          <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
        API Reference
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Basic Endpoint */}
        <ApiEndpointCard
          method="POST"
          endpoint="http://localhost:8000/predict"
          title="Basic Sentiment Analysis"
          requestExample={API_EXAMPLES.basic.request}
          responseExample={API_EXAMPLES.basic.response}
        />

        {/* Detailed Endpoint */}
        <ApiEndpointCard
          method="POST"
          endpoint="http://localhost:8000/predict/detailed"
          title="Detailed Sentiment Analysis"
          requestExample={API_EXAMPLES.detailed.request}
          responseExample={API_EXAMPLES.detailed.response}
        />

        {/* Batch Endpoint */}
        <ApiEndpointCard
          method="POST"
          endpoint="http://localhost:8000/predict/batch"
          title="Batch Processing"
          requestExample={API_EXAMPLES.batch.request}
          responseExample={API_EXAMPLES.batch.response}
          colSpan="md:col-span-2"
        />
      </div>
    </div>
  );
};

export default ApiReference;