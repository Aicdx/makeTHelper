import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import type { ChartPoint, SymbolDetail, SymbolSummary } from '../types'

const API_BASE = 'http://127.0.0.1:18080'
const WS_URL = 'ws://127.0.0.1:18080/ws'

export interface FullSeriesPoint {
  ts: string
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export const useMonitorStore = defineStore('monitor', () => {
  const symbols = ref<SymbolSummary[]>([])
  const selectedSymbol = ref<string | null>(null)
  const selectedDetail = ref<SymbolDetail | null>(null)
  const fullSeries = ref<FullSeriesPoint[] | null>(null)

  const wsConnected = ref(false)
  const lastError = ref<string | null>(null)

  let ws: WebSocket | null = null
  let reconnectTimer: number | null = null

  const selectedSummary = computed(() => {
    if (!selectedSymbol.value) return null
    return symbols.value.find((s) => s.symbol === selectedSymbol.value) || null
  })

  const chartPoints = computed<ChartPoint[]>(() => {
    // Prefer fullSeries if present
    if (fullSeries.value && fullSeries.value.length > 0) {
      return fullSeries.value.map((p) => ({ ts: p.ts, close: p.close }))
    }
    return selectedDetail.value?.close_series || []
  })

  async function fetchSymbols() {
    try {
      lastError.value = null
      const resp = await fetch(`${API_BASE}/api/symbols`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      symbols.value = await resp.json()
    } catch (e: any) {
      lastError.value = `fetchSymbols失败: ${e?.message || String(e)}`
    }
  }

  async function fetchDetail(symbol: string) {
    try {
      lastError.value = null
      const resp = await fetch(`${API_BASE}/api/symbols/${encodeURIComponent(symbol)}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      selectedDetail.value = await resp.json()
    } catch (e: any) {
      lastError.value = `fetchDetail失败: ${e?.message || String(e)}`
    }
  }

  async function fetchFullSeries(symbol: string, limit = 800) {
    try {
      lastError.value = null
      const resp = await fetch(`${API_BASE}/api/full_series/${encodeURIComponent(symbol)}?limit=${limit}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      if (Array.isArray(data)) {
        fullSeries.value = data
      } else {
        // backend may return {error: ...}
        fullSeries.value = null
      }
    } catch (e: any) {
      // fallback to close_series; keep error visible but not fatal
      fullSeries.value = null
      lastError.value = `fetchFullSeries失败: ${e?.message || String(e)}`
    }
  }

  function select(symbol: string) {
    selectedSymbol.value = symbol
    fullSeries.value = null
    void fetchDetail(symbol)
    void fetchFullSeries(symbol)
  }

  function connectWs() {
    if (ws) return

    lastError.value = null
    ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      wsConnected.value = true
      lastError.value = null
    }

    ws.onclose = () => {
      wsConnected.value = false
      ws = null
      scheduleReconnect()
    }

    ws.onerror = () => {
      // onclose will follow
    }

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg?.type === 'update' && Array.isArray(msg.data)) {
          symbols.value = msg.data
        }
      } catch {
        // ignore
      }
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer != null) return
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null
      connectWs()
    }, 2000)
  }

  function disconnectWs() {
    if (reconnectTimer != null) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
    wsConnected.value = false
  }

  return {
    symbols,
    selectedSymbol,
    selectedSummary,
    selectedDetail,
    fullSeries,
    chartPoints,
    wsConnected,
    lastError,
    fetchSymbols,
    fetchDetail,
    fetchFullSeries,
    select,
    connectWs,
    disconnectWs,
  }
})
