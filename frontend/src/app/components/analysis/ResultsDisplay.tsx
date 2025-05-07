import React from 'react';
import { AnalysisResult } from '../../types';
import ResultItem from './ResultItem';

interface ResultsDisplayProps {
  results: AnalysisResult[] | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results }) => {
  if (!results || results.length === 0) {
    return null;
  }

  const positiveCount = results.filter(r => r.result.sentiment === "positive").length;
  const negativeCount = results.filter(r => r.result.sentiment === "negative").length;

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
          <circle cx="12" cy="12" r="10"></circle>
          <path d="M12 16v-4"></path>
          <path d="M12 8h.01"></path>
        </svg>
        Analysis Results
      </h2>

      <div className="space-y-4">
        {results.map((item, index) => (
          <ResultItem key={item.id} item={item} index={index} />
        ))}
      </div>

      {results.length > 1 && (
        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg flex items-start border border-blue-100 dark:border-blue-800/30">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-blue-500 dark:text-blue-400 mr-2 mt-0.5"
          >
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
          </svg>
          <div className="text-xs text-blue-800 dark:text-blue-200">
            <span className="font-medium">Summary:</span>{" "}
            {positiveCount} positive and {negativeCount} negative results
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;