import React from 'react';
import TextInput from './TextInput';
import Button from '../ui/Button';
import { TextInput as TextInputType, AnalysisType } from '../../types';

interface AnalysisFormProps {
  inputs: TextInputType[];
  loading: boolean;
  analysisType: AnalysisType;
  onSubmit: (e: React.FormEvent) => Promise<void>;
  onAddInput: () => void;
  onRemoveInput: (id: number) => void;
  onUpdateInputText: (id: number, text: string) => void;
  onChangeAnalysisType: (type: AnalysisType) => void;
}

const AnalysisForm: React.FC<AnalysisFormProps> = ({
  inputs,
  loading,
  analysisType,
  onSubmit,
  onAddInput,
  onRemoveInput,
  onUpdateInputText,
  onChangeAnalysisType,
}) => {
  return (
    <div className="w-full bg-white dark:bg-black/40 rounded-xl border border-black/[.08] dark:border-white/[.12] p-6 shadow-sm">
      <form onSubmit={onSubmit} className="flex flex-col gap-5">
        <div className="flex justify-between items-center mb-1">
          <h2 className="text-xl font-semibold">Text Analysis</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {inputs.length > 1 ? "Batch Mode" : "Single Input"}
            </span>
            {inputs.length === 1 && (
              <div className="flex items-center bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                <button
                  type="button"
                  onClick={() => onChangeAnalysisType("basic")}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    analysisType === "basic"
                      ? "bg-white dark:bg-gray-700 shadow-sm"
                      : "text-gray-600 dark:text-gray-400"
                  }`}
                >
                  Basic
                </button>
                <button
                  type="button"
                  onClick={() => onChangeAnalysisType("detailed")}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    analysisType === "detailed"
                      ? "bg-white dark:bg-gray-700 shadow-sm"
                      : "text-gray-600 dark:text-gray-400"
                  }`}
                >
                  Detailed
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          {inputs.map((input, index) => (
            <TextInput
              key={input.id}
              id={input.id}
              text={input.text}
              index={index}
              totalInputs={inputs.length}
              onUpdateText={onUpdateInputText}
              onRemove={onRemoveInput}
            />
          ))}
        </div>

        <div className="flex flex-col sm:flex-row gap-3 justify-between mt-2">
          <Button
            type="button"
            onClick={onAddInput}
            className="order-2 sm:order-1"
            icon={
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="12" y1="5" x2="12" y2="19"></line>
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
            }
          >
            Add Another Text Input
          </Button>

          <Button
            type="submit"
            variant="primary"
            isLoading={loading}
            className="order-1 sm:order-2"
            icon={
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
              </svg>
            }
          >
            Analyze Sentiment
          </Button>
        </div>
      </form>
    </div>
  );
};

export default AnalysisForm;