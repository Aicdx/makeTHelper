<script setup lang="ts">
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
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
    out.push({
      name: d.action,
      coord: [p.ts, p.close],
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

const option = computed(() => {
  const pts = normalizedPoints.value
  const x = pts.map((p) => p.ts)
  const y = pts.map((p) => p.close)

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line' },
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
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } },
    },
    series: [
      {
        name: 'close',
        type: 'line',
        data: y,
        showSymbol: false,
        smooth: true,
        lineStyle: { width: 2, color: '#60a5fa' },
        markPoint: {
          data: markPoints.value,
        },
        markLine: {
          symbol: ['none', 'none'],
          data: markLines.value,
          label: {
            color: '#e2e8f0',
            backgroundColor: 'rgba(15,23,42,0.6)',
            padding: [2, 6],
            borderRadius: 4,
          },
        },
      },
    ],
  }
})
</script>

<template>
  <div class="h-64 w-full">
    <VChart class="h-full w-full" :option="option" autoresize />
  </div>
</template>

