import React from 'react';
import { AnalysisResult, InfluentialWord } from '../../types';
import ConfidenceBar from '../ui/ConfidenceBar';

interface ResultItemProps {
  item: AnalysisResult;
  index: number;
}

const ResultItem: React.FC<ResultItemProps> = ({ item, index }) => {
  const { text, result } = item;
  
  return (
    <div className="p-4 border border-black/[.05] dark:border-white/[.08] rounded-lg bg-gray-50 dark:bg-gray-900/50 transition-all hover:shadow-md">
      <div className="flex justify-between items-start">
        <div className="text-xs font-[family-name:var(--font-geist-mono)] text-gray-500 dark:text-gray-400 mb-2 flex-1 truncate pr-4">
          {index + 1}. {text.substring(0, 60)}
          {text.length > 60 ? "..." : ""}
        </div>
        <div
          className={`px-3 py-1 rounded-full text-xs font-medium flex items-center ${
            result.sentiment === "positive"
              ? "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-400"
              : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-400"
          }`}
        >
          {result.sentiment === "positive" ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mr-1"
            >
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
              <line x1="9" y1="9" x2="9.01" y2="9"></line>
              <line x1="15" y1="9" x2="15.01" y2="9"></line>
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mr-1"
            >
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="8" y1="15" x2="16" y2="15"></line>
              <line x1="9" y1="9" x2="9.01" y2="9"></line>
              <line x1="15" y1="9" x2="15.01" y2="9"></line>
            </svg>
          )}
          {result.sentiment === "positive" ? "Positive" : "Negative"}
        </div>
      </div>

      {/* Confidence Bar */}
      {(result.confidence !== undefined || result.confidence_score !== undefined) && (
        <ConfidenceBar result={result} />
      )}

      {/* Influential Words Section */}
      {renderInfluentialWords(result.influential_words)}

      {/* Cleaned Text Section */}
      {result.cleaned_text && (
        <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
          <span className="font-medium">Processed text:</span>{" "}
          {result.cleaned_text}
        </div>
      )}
    </div>
  );
};

// Helper function to render influential words
const renderInfluentialWords = (words?: InfluentialWord[]) => {
  if (!words) return null;
  
  if (words.length === 0) {
    return (
      <div className="mt-3 pt-3 border-t border-black/[.05] dark:border-white/[.08]">
        <div className="text-xs font-medium mb-2">
          Analysis Note:
        </div>
        <div className="text-xs text-gray-600 dark:text-gray-400">
          No specific influential words were identified in this text.
        </div>
      </div>
    );
  }
  
  return (
    <div className="mt-3 pt-3 border-t border-black/[.05] dark:border-white/[.08]">
      <div className="text-xs font-medium mb-2">
        Key Words:
      </div>
      <div className="flex flex-wrap gap-2">
        {words.map((word, wordIndex) => (
          <div
            key={wordIndex}
            className={`px-2 py-1 rounded-md text-xs font-medium ${
              word.sentiment === "positive"
                ? "bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                : word.sentiment === "negative"
                  ? "bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                  : "bg-gray-50 text-gray-700 dark:bg-gray-800 dark:text-gray-400"
            }`}
          >
            {word.word} ({word.score.toFixed(2)})
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultItem;