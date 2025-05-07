// Types for the application

// Input type
export interface TextInput {
    id: number;
    text: string;
  }
  
  // Result types
  export interface BasicResult {
    sentiment: 'positive' | 'negative';
    confidence?: string;
    confidence_score?: number;
  }
  
  export interface DetailedResult extends BasicResult {
    cleaned_text?: string;
    influential_words?: InfluentialWord[];
  }
  
  export interface InfluentialWord {
    word: string;
    score: number;
    sentiment: 'positive' | 'negative' | 'neutral';
  }
  
  export interface AnalysisResult {
    id: number;
    text: string;
    result: DetailedResult;
  }
  
  // Analysis type
  export type AnalysisType = 'basic' | 'detailed';