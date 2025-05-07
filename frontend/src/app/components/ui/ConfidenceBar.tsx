import React from 'react';
import { DetailedResult } from '../../types';
import { getConfidenceValue, getConfidenceDisplay } from '../../utils/formatters';

interface ConfidenceBarProps {
  result: DetailedResult;
}

const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ result }) => {
  if (result.confidence === undefined && result.confidence_score === undefined) {
    return null;
  }

  const confidenceValue = getConfidenceValue(result);
  const confidenceDisplay = getConfidenceDisplay(result);
  const gradientColor = result.sentiment === "positive" 
    ? "bg-gradient-to-r from-green-500 to-green-400" 
    : "bg-gradient-to-r from-red-500 to-red-400";

  return (
    <div className="mt-2">
      <div className="text-xs font-medium">Confidence:</div>
      <div className="mt-1 h-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${gradientColor}`}
          style={{
            width: `${confidenceValue}%`,
          }}
        ></div>
      </div>
      <div className="text-xs mt-1 text-right text-gray-500 dark:text-gray-400">
        {confidenceDisplay}
      </div>
    </div>
  );
};

export default ConfidenceBar;