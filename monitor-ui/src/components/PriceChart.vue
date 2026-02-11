<script setup lang="ts">
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import type { SeriesOption } from 'echarts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkPointComponent,
  MarkLineComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'

import type { ChartPoint, Decision } from '../types'

use([
  CanvasRenderer,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkPointComponent,
  MarkLineComponent,
])

const props = defineProps<{
  points: ChartPoint[]
  decisions?: Decision[]
  basePrice?: number
}>()

const normalizedPoints = computed(() => {
  // de-dup by ts, keep last
  const map = new Map<string, ChartPoint>()
  for (const p of props.points || []) {
    if (!p?.ts) continue
    map.set(p.ts, { ts: p.ts, close: Number(p.close) })
  }
  return Array.from(map.values()).sort((a, b) => (a.ts < b.ts ? -1 : 1))
})

const baseForPct = computed(() => {
  const pts = normalizedPoints.value
  return props.basePrice || (props.points as any)?.[0]?.open || (pts as any)?.[0]?.open || pts[0]?.close
})

const markPoints = computed(() => {
  const pts = normalizedPoints.value
  if (!pts.length) return []

  const byTs = new Map<string, number>()
  pts.forEach((p, idx) => byTs.set(p.ts, idx))

  const out: any[] = []
  for (const d of props.decisions || []) {
    if (!d || !d.action) continue
    if (d.action !== 'BUY_BACK' && d.action !== 'SELL_PART') continue

    // align decision ts to the nearest existing point (best-effort)
    const ts = d.ts
    let idx = byTs.get(ts)
    if (idx === undefined) {
      // fallback: find nearest earlier point
      for (let i = pts.length - 1; i >= 0; i--) {
        if (pts[i].ts <= ts) {
          idx = i
          break
        }
      }
      if (idx === undefined) continue; // Skip if no matching point found
    }

    const p = pts[idx]
    let coordY: number = p.close
    const base = baseForPct.value
    if (Number.isFinite(base) && base && base !== 0) {
      coordY = ((p.close - base) / base) * 100
    }

    out.push({
      name: d.action,
      coord: [p.ts, coordY],
      value: d.action === 'BUY_BACK' ? 'B' : 'S',
      itemStyle: { color: d.action === 'BUY_BACK' ? '#10b981' : '#f43f5e' },
      symbolSize: 36,
      label: { color: '#0b1020', fontWeight: '700' },
    })
  }
  return out
})

const markLines = computed(() => {
  if (!props.decisions || props.decisions.length === 0) return []

  // Find the latest decision with a valid operation plan
  let latestDecision: Decision | null = null
  for (let i = props.decisions.length - 1; i >= 0; i--) {
    const d = props.decisions[i]
    if (d.action === 'BUY_BACK' || d.action === 'SELL_PART') {
      if (d.operation_plan?.target_price > 0) {
        latestDecision = d
        break
      }
    }
  }

  if (!latestDecision) return []

  const plan = latestDecision.operation_plan
  const lines: any[] = []

  // Target price line
  if (plan.target_price > 0) {
    lines.push({
      name: '指导价',
      yAxis: plan.target_price,
      lineStyle: {
        color: latestDecision.action === 'BUY_BACK' ? '#10b981' : '#f43f5e',
        width: 2,
        type: 'solid',
      },
      label: {
        formatter: '指导价: {c}',
        position: 'end',
        color: '#fff',
        backgroundColor: latestDecision.action === 'BUY_BACK' ? '#10b981' : '#f43f5e',
        padding: [2, 5],
        borderRadius: 3,
        fontWeight: 'bold',
      },
    })
  }

  // Stop price line
  if (plan.stop_price > 0) {
    lines.push({
      name: '止损价',
      yAxis: plan.stop_price,
      lineStyle: {
        color: '#f59e0b',
        width: 1,
        type: 'dashed',
      },
      label: {
        formatter: '止损: {c}',
        position: 'end',
        color: '#111827',
        backgroundColor: '#f59e0b',
        padding: [2, 5],
        borderRadius: 3,
        fontWeight: 'bold',
      },
    })
  }

  return lines
})

function _calcMA(values: (number | null)[], period: number): Array<number | null> {
  const out: Array<number | null> = new Array(values.length).fill(null)
  if (!period || period <= 1) return values.map((v) => (v !== null ? v : null))

  let sum = 0
  let count = 0
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (v !== null) {
      sum += v
      count++
    }

    if (i >= period) {
      const prevV = values[i - period]
      if (prevV !== null) {
        sum -= prevV
        count--
      }
    }

    if (count === period) {
      out[i] = sum / period
    }
  }
  return out
}

const option = computed(() => {
  const pts = normalizedPoints.value
  const x = pts.map((p) => p.ts)
  const close = pts.map((p) => p.close)

  const base = baseForPct.value

  const pct = close.map((v) => {
    if (v === null || v === undefined || !base) return null
    return ((v - base) / base) * 100
  })

  const ma5 = _calcMA(pct, 5)
  const ma10 = _calcMA(pct, 10)
  const ma20 = _calcMA(pct, 20)

  const series: SeriesOption[] = [
    {
      name: '涨跌幅(%)',
      type: 'line',
      data: pct,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 2, color: '#60a5fa' },
      markPoint: {
        data: markPoints.value,
      },
      markLine: {
        symbol: ['none', 'none'],
        data: [
          {
            name: '0轴',
            yAxis: 0,
            lineStyle: {
              color: 'rgba(148,163,184,0.45)',
              width: 1,
              type: 'dashed',
            },
            label: {
              formatter: '0轴',
              position: 'end',
              color: '#94a3b8',
              backgroundColor: 'transparent',
            },
          },
        ],
        label: {
          color: '#e2e8f0',
          backgroundColor: 'rgba(15,23,42,0.6)',
          padding: [2, 6],
          borderRadius: 4,
        },
      },
    },
    {
      name: 'MA5',
      type: 'line',
      data: ma5,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 1, color: 'rgba(250,204,21,0.9)' },
      emphasis: { focus: 'series' },
    },
    {
      name: 'MA10',
      type: 'line',
      data: ma10,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 1, color: 'rgba(248,113,113,0.85)' },
      emphasis: { focus: 'series' },
    },
    {
      name: 'MA20',
      type: 'line',
      data: ma20,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 1, color: 'rgba(52,211,153,0.85)' },
      emphasis: { focus: 'series' },
    },
  ]

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line' },
      formatter: (params: any) => {
        const arr = Array.isArray(params) ? params : []
        const ts = arr[0]?.axisValueLabel ?? arr[0]?.axisValue ?? ''

        const byName = new Map<string, any>()
        for (const p of arr) byName.set(String(p?.seriesName || ''), p)

        const closeVal = close[Number(arr[0]?.dataIndex ?? 0)]
        const pctVal = Number(byName.get('涨跌幅(%)')?.data)

        const pctText = Number.isFinite(pctVal) ? `${pctVal > 0 ? '+' : ''}${pctVal.toFixed(2)}%` : '-'
        const closeText = Number.isFinite(closeVal) ? closeVal.toFixed(2) : '-'

        const ma5v = Number(byName.get('MA5')?.data)
        const ma10v = Number(byName.get('MA10')?.data)
        const ma20v = Number(byName.get('MA20')?.data)

        const ma5t = Number.isFinite(ma5v) ? `${ma5v > 0 ? '+' : ''}${ma5v.toFixed(2)}%` : '-'
        const ma10t = Number.isFinite(ma10v) ? `${ma10v > 0 ? '+' : ''}${ma10v.toFixed(2)}%` : '-'
        const ma20t = Number.isFinite(ma20v) ? `${ma20v > 0 ? '+' : ''}${ma20v.toFixed(2)}%` : '-'

        const baseText = Number.isFinite(base) ? String(Number(base).toFixed(2)) : '-'

        return `${ts}<br/>open: ${baseText}<br/>close: ${closeText}<br/>涨跌幅: ${pctText}<br/>MA5: ${ma5t} &nbsp; MA10: ${ma10t} &nbsp; MA20: ${ma20t}`
      },
    },
    grid: { left: 40, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'category',
      data: x,
      axisLabel: {
        color: '#94a3b8',
        formatter: (value: string) => {
          try {
            const d = new Date(value)
            return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
          } catch (e) {
            return value.slice(11, 16)
          }
        },
      },
      axisLine: { lineStyle: { color: '#334155' } },
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLabel: {
        color: '#94a3b8',
        formatter: (v: number) => `${Number(v).toFixed(2)}%`,
      },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } },
    },
    legend: {
      top: 0,
      textStyle: { color: '#94a3b8' },
    },
    series,
  }
})
</script>

<template>
  <div class="h-64 w-full">
    <VChart class="h-full w-full" :option="option" autoresize />
  </div>
</template>

