import { useState } from "react";
import { TextInput, AnalysisResult, AnalysisType } from "../types";
import { API_URL } from "../utils/constants";

export const useAnalysis = () => {
  // State for text inputs and results
  const [inputs, setInputs] = useState<TextInput[]>([{ id: 1, text: "" }]);
  const [results, setResults] = useState<AnalysisResult[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisType, setAnalysisType] = useState<AnalysisType>("basic");

  // Add a new text input field
  const addInput = () => {
    const newId =
      inputs.length > 0 ? Math.max(...inputs.map((i) => i.id)) + 1 : 1;
    setInputs([...inputs, { id: newId, text: "" }]);
  };

  // Remove a text input field
  const removeInput = (id: number) => {
    if (inputs.length > 1) {
      setInputs(inputs.filter((input) => input.id !== id));
      // Clear results when removing inputs
      setResults(null);
    }
  };

  // Update text in an input field
  const updateInputText = (id: number, text: string) => {
    setInputs(
      inputs.map((input) => (input.id === id ? { ...input, text } : input))
    );
    // Clear results when inputs change
    setResults(null);
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Filter out empty inputs
      const validInputs = inputs.filter((input) => input.text.trim());

      if (validInputs.length === 0) {
        throw new Error("Please enter at least one text to analyze");
      }

      let analysisResults: AnalysisResult[] = [];

      // Use batch endpoint for multiple inputs
      if (validInputs.length > 1) {
        const response = await fetch(API_URL.batch, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(
            validInputs.map((input) => ({ text: input.text }))
          ),
        });

        const data = await response.json();

        // Match results with input texts
        analysisResults = validInputs.map((input, index) => ({
          id: input.id,
          text: input.text,
          result: data[index],
        }));
      }
      // Use single endpoint for one input
      else {
        const endpoint =
          analysisType === "detailed" ? API_URL.detailed : API_URL.basic;
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: validInputs[0].text }),
        });

        const data = await response.json();
        analysisResults = [
          {
            id: validInputs[0].id,
            text: validInputs[0].text,
            result: data,
          },
        ];
      }

      setResults(analysisResults);
    } catch (error) {
      console.error("Error:", error);
      alert(
        error instanceof Error 
          ? error.message 
          : "An error occurred while analyzing the text. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return {
    inputs,
    results,
    loading,
    analysisType,
    setAnalysisType,
    addInput,
    removeInput,
    updateInputText,
    handleSubmit,
  };
};