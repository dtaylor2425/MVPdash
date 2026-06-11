'use client'

import { useState, useEffect } from 'react'
import { getPortfolioSummary } from '@/lib/api'

function CurveChart({ curve, period }) {
  if (!curve || !Array.isArray(curve) || curve.length < 3) return <div style={{ width: '100%', height: 200, background: '#0b1018', borderRadius: 6 }} />

  // Filter curve by period
  var filtered = curve
  if (period !== 'all' && curve.length > 0) {
    var lastDate = new Date(curve[curve.length - 1].date)
    var cutoff = new Date(lastDate)
    if (period === '1m') cutoff.setDate(cutoff.getDate() - 30)
    else if (period === '3m') cutoff.setMonth(cutoff.getMonth() - 3)
    else if (period === '6m') cutoff.setMonth(cutoff.getMonth() - 6)
    else if (period === 'ytd') { cutoff = new Date(lastDate.getFullYear(), 0, 1) }
    else if (period === '1y') cutoff.setFullYear(cutoff.getFullYear() - 1)
    else if (period === '2y') cutoff.setFullYear(cutoff.getFullYear() - 2)
    filtered = curve.filter(function(p) { return new Date(p.date) >= cutoff })
  }
  if (filtered.length < 2) filtered = curve

  // Normalize to period start
  var startP = filtered[0].portfolio || 1
  var startS = filtered[0].spy || 1
  var startB = filtered[0].bench_60_40 || 1

  var w = 400, h = 180
  var allVals = []
  filtered.forEach(function(p) {
    allVals.push((p.portfolio || startP) / startP)
    allVals.push((p.spy || startS) / startS)
    allVals.push((p.bench_60_40 || startB) / startB)
  })
  var mn = Math.min.apply(null, allVals), mx = Math.max.apply(null, allVals), rg = mx - mn || 0.01

  function makeLine(key, startVal) {
    return filtered.map(function(p, i) {
      var v = ((p[key] || startVal) / startVal)
      return ((i / (filtered.length - 1)) * w) + ',' + (h - 8 - ((v - mn) / rg) * (h - 16))
    }).join(' ')
  }

  var portfolioLine = makeLine('portfolio', startP)
  var spyLine = makeLine('spy', startS)
  var benchLine = makeLine('bench_60_40', startB)

  var gid = 'pc' + Math.random().toString(36).slice(2, 6)

  return (
    <svg viewBox={'0 0 ' + w + ' ' + h} style={{ width: '100%', height: 200, display: 'block' }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#d4a017" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#d4a017" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* 60/40 benchmark */}
      <polyline points={benchLine} fill="none" stroke="#3a4555" strokeWidth="1" strokeDasharray="4 3" />
      {/* SPY */}
      <polyline points={spyLine} fill="none" stroke="#22c55e" strokeWidth="1.2" strokeOpacity="0.6" />
      {/* Portfolio */}
      <polygon points={portfolioLine + ' ' + w + ',' + h + ' 0,' + h} fill={'url(#' + gid + ')'} />
      <polyline points={portfolioLine} fill="none" stroke="#d4a017" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

export default function PortfolioCard() {
  var [data, setData] = useState(null)
  var [period, setPeriod] = useState('ytd')
  var [loading, setLoading] = useState(true)

  useEffect(function() {
    getPortfolioSummary().then(function(d) {
      if (d) setData(d)
      setLoading(false)
    }).catch(function() { setLoading(false) })
  }, [])

  if (loading) {
    return (
      <div style={{ background: '#0b1018', border: '1px solid #1e2a38', borderRadius: 12, padding: '20px', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="ping" style={{ width: 6, height: 6, borderRadius: '50%', background: '#d4a017' }} />
          <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, color: '#5a6a7a' }}>Loading portfolio...</span>
        </div>
      </div>
    )
  }

  if (!data) return null

  var returns = data.returns || {}
  var spyReturns = data.spy_returns || {}
  var currentReturn = returns[period] !== null && returns[period] !== undefined ? returns[period] : returns['ytd']
  var spyReturn = spyReturns[period] !== null && spyReturns[period] !== undefined ? spyReturns[period] : spyReturns['ytd']
  var weights = data.current_weights || {}
  var lastReb = data.last_rebalance

  var periods = ['1m', '3m', '6m', 'ytd', '1y', '2y']

  // Top holdings sorted by weight
  var holdings = Object.entries(weights).sort(function(a, b) { return b[1] - a[1] })

  return (
    <a href="/dashboard/signals" style={{ textDecoration: 'none', display: 'block' }}>
      <div style={{
        background: '#0b1018', border: '1px solid #1e2a38', borderRadius: 12,
        padding: '20px', marginBottom: 14, transition: 'border-color 0.2s',
        cursor: 'pointer',
      }}
        onMouseEnter={function(e) { e.currentTarget.style.borderColor = '#d4a01744' }}
        onMouseLeave={function(e) { e.currentTarget.style.borderColor = '#1e2a38' }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <div style={{ width: 3, height: 12, borderRadius: 1, background: '#d4a017' }} />
              <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: '0.85rem', color: '#e0e4ea', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Macro Engine Portfolio</span>
            </div>
            <div style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#5a6a7a', letterSpacing: '0.06em', marginLeft: 11 }}>
              REGIME-ADAPTIVE \u00B7 WEEKLY REBALANCE \u00B7 LIVE TRACK RECORD
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontFamily: 'DM Mono, monospace', fontSize: 22, fontWeight: 700, color: typeof currentReturn === 'number' && currentReturn >= 0 ? '#22c55e' : '#ef4444' }}>
              {typeof currentReturn === 'number' ? (currentReturn >= 0 ? '+' : '') + currentReturn.toFixed(1) + '%' : '\u2014'}
            </div>
            <div style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#5a6a7a', marginTop: 1 }}>
              vs SPY {typeof spyReturn === 'number' ? (spyReturn >= 0 ? '+' : '') + spyReturn.toFixed(1) + '%' : '\u2014'}
            </div>
          </div>
        </div>

        {/* Period toggles */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 12 }} onClick={function(e) { e.preventDefault(); e.stopPropagation() }}>
          {periods.map(function(p) {
            var active = period === p
            return (
              <button key={p} onClick={function(e) { e.preventDefault(); e.stopPropagation(); setPeriod(p) }} style={{
                padding: '3px 8px', borderRadius: 4, border: 'none',
                background: active ? '#d4a01720' : 'transparent',
                color: active ? '#d4a017' : '#3a4555',
                fontFamily: 'DM Mono, monospace', fontSize: 9, fontWeight: active ? 600 : 400,
                cursor: 'pointer', transition: 'all 0.15s',
                textTransform: 'uppercase',
              }}>{p}</button>
            )
          })}
        </div>

        {/* Chart */}
        <div style={{ marginBottom: 12 }}>
          <CurveChart curve={data.curve} period={period} />
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 12, height: 2, background: '#d4a017', borderRadius: 1 }} />
            <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#6a7a8a' }}>Portfolio</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 12, height: 2, background: '#22c55e', borderRadius: 1, opacity: 0.6 }} />
            <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#6a7a8a' }}>SPY</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 12, height: 2, background: '#3a4555', borderRadius: 1, borderTop: '1px dashed #3a4555' }} />
            <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#6a7a8a' }}>60/40</span>
          </div>
          {typeof data.max_drawdown === 'number' && (
            <div style={{ marginLeft: 'auto' }}>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#3a4555' }}>Max DD: </span>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#ef4444' }}>{data.max_drawdown.toFixed(1)}%</span>
            </div>
          )}
        </div>

        {/* Current allocation */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
          {holdings.map(function(entry) {
            var ticker = entry[0], weight = entry[1]
            if (weight < 0.02) return null
            var colors = { SPY: '#22c55e', QQQ: '#7c3aed', SMH: '#3b82f6', GLD: '#d4a017', SLV: '#94a3b8', TLT: '#0ea5e9', HYG: '#f59e0b', SHY: '#6b7280' }
            var c = colors[ticker] || '#5a6a7a'
            return (
              <div key={ticker} style={{ flex: weight, height: 6, background: c, borderRadius: 3, minWidth: 4, position: 'relative' }} title={ticker + ' ' + (weight * 100).toFixed(0) + '%'} />
            )
          })}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
          {holdings.map(function(entry) {
            var ticker = entry[0], weight = entry[1]
            if (weight < 0.02) return null
            var colors = { SPY: '#22c55e', QQQ: '#7c3aed', SMH: '#3b82f6', GLD: '#d4a017', SLV: '#94a3b8', TLT: '#0ea5e9', HYG: '#f59e0b', SHY: '#6b7280' }
            var c = colors[ticker] || '#5a6a7a'
            return (
              <div key={ticker} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <div style={{ width: 6, height: 6, borderRadius: 2, background: c }} />
                <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#5a6a7a' }}>{ticker}</span>
                <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#6a7a8a', fontWeight: 600 }}>{(weight * 100).toFixed(0)}%</span>
              </div>
            )
          })}
        </div>

        {/* Last rebalance */}
        {lastReb && (
          <div style={{ padding: '10px 12px', background: '#0d1420', border: '1px solid #1e2a38', borderRadius: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#d4a017', textTransform: 'uppercase', letterSpacing: '0.08em', fontWeight: 600 }}>Last Rebalance</span>
              <span style={{ fontFamily: 'DM Mono, monospace', fontSize: 8, color: '#3a4555' }}>{lastReb.date}</span>
            </div>
            <div style={{ fontFamily: 'DM Mono, monospace', fontSize: 10, color: '#6a7a8a', lineHeight: 1.5 }}>{lastReb.rationale}</div>
          </div>
        )}
      </div>
    </a>
  )
}