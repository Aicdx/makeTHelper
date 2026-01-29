<script setup lang="ts">
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkPointComponent } from 'echarts/components'
import VChart from 'vue-echarts'

import type { ChartPoint, Decision } from '../types'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, MarkPointComponent])

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
      if (idx === undefined) idx = pts.length - 1
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
      axisLabel: { color: '#94a3b8', formatter: (v: string) => v.slice(11, 16) },
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

