<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

import PriceChart from '../components/PriceChart.vue'
import { useMonitorStore } from '../stores/monitor'

const store = useMonitorStore()

const symbolFilter = ref('')
const actionFilter = ref<string>('')

const actions = ['BUY_BACK', 'SELL_PART', 'HOLD_POSITION', 'CANCEL_PLAN', 'INFO']

const filteredSymbols = computed(() => {
  const kw = symbolFilter.value.trim().toLowerCase()
  const act = actionFilter.value
  return store.symbols
    .filter((s) => {
      if (!kw) return true
      return s.symbol.toLowerCase().includes(kw) || (s.name || '').toLowerCase().includes(kw)
    })
    .filter((s) => {
      if (!act) return true
      return (s.latest_action || '') === act
    })
    .sort((a, b) => (b.latest_confidence || 0) - (a.latest_confidence || 0))
})

const drawerOpen = computed(() => !!store.selectedSymbol)

function closeDrawer() {
  store.selectedSymbol = null
  store.selectedDetail = null
}

const latestDecision = computed(() => store.selectedDetail?.decisions?.[0] || null)

function _safeNum(x: unknown, d = 0): number {
  const n = Number(x)
  return Number.isFinite(n) ? n : d
}

const targetDistance = computed(() => {
  const price = _safeNum(store.selectedSummary?.close, 0)
  const target = _safeNum(latestDecision.value?.operation_plan?.target_price, 0)
  if (!target || target <= 0) return null
  const diff = price - target
  const pct = (diff / target) * 100
  return { price, target, diff, pct }
})

const stopDistance = computed(() => {
  const price = _safeNum(store.selectedSummary?.close, 0)
  const stop = _safeNum(latestDecision.value?.operation_plan?.stop_price, 0)
  if (!stop || stop <= 0) return null
  const diff = price - stop
  const pct = (diff / stop) * 100
  return { price, stop, diff, pct }
})

const targetDistanceClass = computed(() => {
  const d = targetDistance.value
  if (!d) return 'text-slate-300'
  // 越接近 0 越“快到了”，用更亮的颜色提示
  if (Math.abs(d.pct) <= 0.3) return 'text-emerald-300'
  if (Math.abs(d.pct) <= 1.0) return 'text-emerald-200'
  return 'text-slate-300'
})

const stopDistanceClass = computed(() => {
  const d = stopDistance.value
  if (!d) return 'text-slate-300'
  // 越接近止损(0)越危险，显示更红
  if (Math.abs(d.pct) <= 0.3) return 'text-rose-300'
  if (Math.abs(d.pct) <= 1.0) return 'text-rose-200'
  return 'text-slate-300'
})

let detailTimer: number | null = null

function startDetailAutoRefresh() {
  if (detailTimer != null) return
  detailTimer = window.setInterval(() => {
    if (store.selectedSymbol) {
      void store.fetchDetail(store.selectedSymbol)
    }
  }, 8000)
}

function stopDetailAutoRefresh() {
  if (detailTimer != null) {
    window.clearInterval(detailTimer)
    detailTimer = null
  }
}

onMounted(async () => {
  await store.fetchSymbols()
  store.connectWs()
})

onUnmounted(() => {
  store.disconnectWs()
  stopDetailAutoRefresh()
})

watch(
  () => store.selectedSymbol,
  (sym) => {
    if (sym) startDetailAutoRefresh()
    else stopDetailAutoRefresh()
  },
)
</script>

<template>
  <div class="min-h-screen bg-slate-950 text-slate-100">
    <header class="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/80 backdrop-blur">
      <div class="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div>
          <h1 class="text-xl font-semibold">做T监控台</h1>
          <div class="mt-1 text-xs text-slate-400">
            后端：<span class="font-mono">http://127.0.0.1:18080</span>
            <span class="mx-2">|</span>
            WS：<span class="font-mono">ws://127.0.0.1:18080/ws</span>
          </div>
        </div>
        <div class="flex items-center gap-3 text-sm">
          <span
            class="inline-flex items-center rounded-full px-2 py-1"
            :class="store.wsConnected ? 'bg-emerald-900/40 text-emerald-200' : 'bg-rose-900/40 text-rose-200'"
          >
            {{ store.wsConnected ? 'WS已连接' : 'WS断开' }}
          </span>
        </div>
      </div>
    </header>

    <main class="mx-auto max-w-7xl px-4 py-6">
      <div
        v-if="store.lastError"
        class="mb-4 rounded-lg border border-rose-800 bg-rose-950/40 p-3 text-sm text-rose-200"
      >
        {{ store.lastError }}
      </div>

      <div class="mb-4 grid gap-3 md:grid-cols-3">
        <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
          <div class="text-xs text-slate-400">搜索（symbol/名称）</div>
          <input
            v-model="symbolFilter"
            class="mt-2 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-slate-500"
            placeholder="例如：sz300308 / 中际"
          />
        </div>
        <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
          <div class="text-xs text-slate-400">按 action 过滤</div>
          <select
            v-model="actionFilter"
            class="mt-2 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm outline-none focus:border-slate-500"
          >
            <option value="">全部</option>
            <option v-for="a in actions" :key="a" :value="a">{{ a }}</option>
          </select>
        </div>
        <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
          <div class="text-xs text-slate-400">数据条数</div>
          <div class="mt-2 text-sm">
            当前：<span class="font-mono">{{ store.symbols.length }}</span>
            <span class="mx-2">|</span>
            过滤后：<span class="font-mono">{{ filteredSymbols.length }}</span>
          </div>
          <button
            class="mt-2 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm hover:bg-slate-900"
            @click="store.fetchSymbols()"
          >
            手动刷新
          </button>
        </div>
      </div>

      <div class="overflow-hidden rounded-xl border border-slate-800">
        <table class="w-full border-collapse text-sm">
          <thead class="bg-slate-900/60 text-slate-300">
            <tr>
              <th class="px-3 py-3 text-left font-medium">标的</th>
              <th class="px-3 py-3 text-left font-medium">最新价</th>
              <th class="px-3 py-3 text-left font-medium">涨跌幅</th>
              <th class="px-3 py-3 text-left font-medium">action / conf</th>
              <th class="px-3 py-3 text-left font-medium">总仓 / t_share</th>
              <th class="px-3 py-3 text-left font-medium">状态</th>
              <th class="px-3 py-3 text-left font-medium">时间</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="s in filteredSymbols"
              :key="s.symbol"
              class="cursor-pointer border-t border-slate-800 hover:bg-slate-900/50"
              @click="store.select(s.symbol)"
            >
              <td class="px-3 py-3">
                <div class="font-mono">{{ s.symbol }}</div>
                <div class="text-xs text-slate-400">{{ s.name }}</div>
              </td>
              <td class="px-3 py-3 font-mono">{{ s.close?.toFixed(2) }}</td>
              <td class="px-3 py-3 font-mono">
                <span v-if="s.change_pct !== undefined" :class="s.change_pct > 0 ? 'text-rose-400' : s.change_pct < 0 ? 'text-emerald-400' : 'text-slate-400'">
                  {{ s.change_pct > 0 ? '+' : '' }}{{ s.change_pct.toFixed(2) }}%
                </span>
                <span v-else class="text-slate-500">-</span>
              </td>
              <td class="px-3 py-3">
                <div class="font-mono">{{ s.latest_action || '-' }}</div>
                <div class="text-xs text-slate-400">conf={{ (s.latest_confidence ?? 0).toFixed(2) }}</div>
              </td>
              <td class="px-3 py-3 font-mono">
                {{ (s.total_share ?? 1).toFixed(1) }} / {{ (s.t_share ?? 0).toFixed(1) }}
              </td>
              <td class="px-3 py-3 font-mono text-xs">{{ s.position_state }}</td>
              <td class="px-3 py-3 font-mono text-xs">{{ s.ts }}</td>
            </tr>
            <tr v-if="filteredSymbols.length === 0">
              <td class="px-3 py-6 text-center text-slate-400" colspan="6">无数据</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Drawer -->
      <div v-if="drawerOpen" class="fixed inset-0 z-30">
        <div class="absolute inset-0 bg-black/60" @click="closeDrawer" />
        <div
          class="absolute right-0 top-0 h-full w-full max-w-2xl overflow-y-auto border-l border-slate-800 bg-slate-950"
        >
          <div class="flex items-center justify-between border-b border-slate-800 px-4 py-4">
            <div>
              <div class="font-mono text-sm text-slate-300">{{ store.selectedSymbol }}</div>
              <div class="text-lg font-semibold">
                {{ store.selectedDetail?.name || store.selectedSummary?.name }}
              </div>
            </div>
            <button
              class="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm hover:bg-slate-900"
              @click="closeDrawer"
            >
              关闭
            </button>
          </div>

          <div class="p-4">
            <div class="grid gap-3 md:grid-cols-2">
              <div class="rounded-lg border border-slate-800 bg-slate-900/30 p-3">
                <div class="text-xs text-slate-400">最新</div>
                <div class="mt-2 font-mono text-sm">
                  close={{ store.selectedSummary?.close?.toFixed(2) }}
                </div>
                <div class="mt-1 font-mono text-sm">
                  action={{ store.selectedSummary?.latest_action }} conf={{ (store.selectedSummary?.latest_confidence ?? 0).toFixed(2) }}
                </div>
                <div class="mt-1 font-mono text-sm">
                  total={{ (store.selectedSummary?.total_share ?? 1).toFixed(1) }} t_share={{ (store.selectedSummary?.t_share ?? 0).toFixed(1) }}
                </div>
              </div>
              <div class="rounded-lg border border-slate-800 bg-slate-900/30 p-3">
                <div class="text-xs text-slate-400">操作计划（最新一条）</div>
                <div class="mt-2 text-sm text-slate-200">
                  <div v-if="latestDecision">
                    <div class="font-mono">target={{ latestDecision.operation_plan?.target_price }}</div>
                    <div class="font-mono">stop={{ latestDecision.operation_plan?.stop_price }}</div>
                    <div class="font-mono">share={{ latestDecision.operation_plan?.suggested_share }}</div>
                    <div class="font-mono">window={{ latestDecision.operation_plan?.time_window }}</div>

                    <div v-if="targetDistance" class="mt-2 text-xs text-slate-400">
                      距离 target：
                      <span class="font-mono" :class="targetDistanceClass">
                        {{ targetDistance.diff.toFixed(2) }} ({{ targetDistance.pct.toFixed(2) }}%)
                      </span>
                    </div>
                    <div v-if="stopDistance" class="text-xs text-slate-400">
                      距离 stop：
                      <span class="font-mono" :class="stopDistanceClass">
                        {{ stopDistance.diff.toFixed(2) }} ({{ stopDistance.pct.toFixed(2) }}%)
                      </span>
                    </div>
                  </div>
                  <div v-else class="text-slate-400">暂无</div>
                </div>
              </div>
            </div>

            <div class="mt-4 rounded-lg border border-slate-800 bg-slate-900/30 p-3">
              <div class="text-xs text-slate-400">最近决策（最多50条）</div>
              <div class="mt-3 space-y-3">
                <div
                  v-for="(d, idx) in store.selectedDetail?.decisions || []"
                  :key="idx"
                  class="rounded-md border border-slate-800 bg-slate-950 p-3"
                >
                  <div class="flex items-center justify-between">
                    <div class="font-mono text-sm">{{ d.action }} (conf={{ d.confidence.toFixed(2) }})</div>
                  </div>
                  <div class="mt-2 text-xs text-slate-400">reasons</div>
                  <ul class="mt-1 list-disc pl-5 text-sm text-slate-200">
                    <li v-for="(r, i) in d.reasons" :key="i">{{ r }}</li>
                  </ul>
                  <div class="mt-2 text-xs text-slate-400">risks</div>
                  <ul class="mt-1 list-disc pl-5 text-sm text-slate-200">
                    <li v-for="(r, i) in d.risks" :key="i">{{ r }}</li>
                  </ul>
                  <div class="mt-2 text-xs text-slate-400">next_decision_point</div>
                  <ul class="mt-1 list-disc pl-5 text-sm text-slate-200">
                    <li v-for="(r, i) in d.next_decision_point" :key="i">{{ r }}</li>
                  </ul>
                </div>
              </div>
            </div>

            <div class="mt-4 rounded-lg border border-slate-800 bg-slate-900/30 p-3">
              <div class="flex items-center justify-between">
                <div class="text-xs text-slate-400">价格序列（close_series）</div>
                <div class="text-xs text-slate-500">
                  标注：<span class="text-emerald-300">B=BUY_BACK</span> / <span class="text-rose-300">S=SELL_PART</span>
                </div>
              </div>
              <div class="mt-3">
                <PriceChart
                  v-if="store.selectedDetail"
                  :points="store.chartPoints"
                  :decisions="store.selectedDetail.decisions || []"
                  :base-price="store.selectedSummary?.open"
                />
                <div v-else class="text-xs text-slate-400">暂无图表数据</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>
