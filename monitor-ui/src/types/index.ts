// Types for the monitoring dashboard, matching the backend API

export interface SymbolSummary {
  symbol: string
  name: string
  ts: string // ISO 8601 string
  close: number
  position_state: string
  t_share: number
  total_share: number
  latest_action?: string
  latest_confidence?: number
}

export interface OperationPlan {
  target_price: number
  stop_price: number
  suggested_share: string
  time_window: string
}

export interface Decision {
  ts: string // ISO 8601 string
  action: string
  confidence: number
  reasons: string[]
  risks: string[]
  operation_plan: OperationPlan
  next_decision_point: string[]
}

export interface ChartPoint {
  ts: string // ISO 8601 string
  close: number
}

export interface SymbolDetail {
  symbol: string
  name: string
  close_series: ChartPoint[]
  decisions: Decision[]
}

