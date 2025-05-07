"use client";

import React from 'react';
import Header from '@/app/components/layout/Header';
import Footer from '@/app/components/layout/Footer';
import AnalysisForm from '@/app/components/analysis/AnalysisForm';
import ResultsDisplay from '@/app/components/analysis/ResultsDisplay';
import ApiReference from '@/app/components/api-reference/ApiReference';
import { useAnalysis } from '@/app/hooks/useAnalysis';

export default function Home() {
  const {
    inputs,
    results,
    loading,
    analysisType,
    setAnalysisType,
    addInput,
    removeInput,
    updateInputText,
    handleSubmit,
  } = useAnalysis();

  return (
    <div className="grid grid-rows-[auto_1fr_auto] items-center justify-items-center min-h-screen p-6 pb-12 gap-8 sm:p-12 font-[family-name:var(--font-geist-sans)] bg-[#fafafa] dark:bg-[#080808]">
      <Header />

      <main className="flex flex-col gap-6 items-center w-full max-w-4xl">
        {/* Analysis Form */}
        <AnalysisForm
          inputs={inputs}
          loading={loading}
          analysisType={analysisType}
          onSubmit={handleSubmit}
          onAddInput={addInput}
          onRemoveInput={removeInput}
          onUpdateInputText={updateInputText}
          onChangeAnalysisType={setAnalysisType}
        />

        {/* Results */}
        <ResultsDisplay results={results} />

        {/* API Information */}
        <ApiReference />
      </main>

      <Footer />
    </div>
  );
}