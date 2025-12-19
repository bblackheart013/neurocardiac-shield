'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Activity, Brain, Heart, Cpu, Database,
  Monitor, ChevronDown, ArrowRight, Check,
  Zap, Shield, GitBranch, FileText, Users,
  Layers, BarChart3, Settings, ExternalLink
} from 'lucide-react'

// Navigation
function Navigation() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const links = [
    { href: '#architecture', label: 'Architecture' },
    { href: '#data', label: 'Data' },
    { href: '#intelligence', label: 'Intelligence' },
    { href: '#device', label: 'Connect' },
    { href: '#verification', label: 'Verify' },
  ]

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled ? 'bg-nc-bg/95 backdrop-blur-lg border-b border-white/5' : ''
    }`}>
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="font-mono text-sm text-nc-accent border border-nc-accent px-2 py-1 rounded">
            NC
          </div>
          <span className="text-sm font-medium hidden sm:block">NeuroCardiac Shield</span>
        </div>
        <div className="flex items-center gap-6">
          {links.map(link => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm text-white/60 hover:text-white transition-colors hidden md:block"
            >
              {link.label}
            </a>
          ))}
          <a
            href="#credits"
            className="text-sm px-4 py-2 border border-white/10 rounded-lg hover:border-nc-accent/50 transition-colors"
          >
            About
          </a>
        </div>
      </div>
    </nav>
  )
}

// Hero Section
function Hero() {
  return (
    <section className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden px-6">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-nc-accent/5 via-transparent to-transparent" />

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-4xl relative z-10"
      >
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-block font-mono text-xs text-white/40 border border-white/10 px-4 py-2 rounded-full mb-8"
        >
          NYU TANDON · ECE-GY 9953 · FALL 2025
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-light tracking-tight mb-6">
          <span className="block">NeuroCardiac</span>
          <span className="block gradient-text font-normal">Shield</span>
        </h1>

        <p className="text-xl text-white/60 max-w-2xl mx-auto mb-12 leading-relaxed">
          An integrated brain-heart monitoring system that combines EEG and ECG analysis
          for early physiological risk detection.
        </p>

        <div className="flex items-center justify-center gap-4 text-sm text-white/40">
          <span>Mohd Sarfaraz Faiyaz</span>
          <span className="w-1 h-1 rounded-full bg-white/20" />
          <span>Vaibhav D. Chandgir</span>
        </div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      >
        <ChevronDown className="w-5 h-5 text-white/20 animate-bounce" />
        <span className="font-mono text-xs text-white/20 uppercase tracking-widest">Scroll</span>
      </motion.div>
    </section>
  )
}

// Architecture Section
function Architecture() {
  const layers = [
    {
      num: '04',
      title: 'Presentation',
      desc: 'Real-time dashboard with EEG waveforms, ECG traces, and risk visualization',
      tech: 'Streamlit · Plotly · WebSocket',
      color: 'nc-accent',
      icon: Monitor,
      metrics: [{ label: 'Latency', value: '<16ms' }, { label: 'Refresh', value: '10 Hz' }]
    },
    {
      num: '03',
      title: 'Application',
      desc: 'REST API endpoints for data ingestion, inference triggering, and streaming',
      tech: 'FastAPI · Pydantic · Uvicorn',
      color: 'nc-hrv',
      icon: Cpu,
      metrics: [{ label: 'Endpoints', value: '5' }, { label: 'Buffer', value: '1000 pkts' }]
    },
    {
      num: '02',
      title: 'Domain',
      desc: 'Signal processing pipelines and ML ensemble for risk classification',
      tech: 'NumPy · SciPy · XGBoost · TensorFlow',
      color: 'nc-eeg',
      icon: Brain,
      metrics: [{ label: 'Features', value: '76' }, { label: 'Models', value: '2' }]
    },
    {
      num: '01',
      title: 'Acquisition',
      desc: 'Device adapters for simulated, BLE, and serial sensor connections',
      tech: 'C Firmware · Python Adapters',
      color: 'nc-ecg',
      icon: Activity,
      metrics: [{ label: 'Channels', value: '11' }, { label: 'Rate', value: '250 Hz' }]
    },
  ]

  return (
    <section id="architecture" className="py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="font-mono text-nc-accent text-sm">01</span>
          <h2 className="section-title mt-4 mb-4">The Living Architecture</h2>
          <p className="text-white/60 max-w-xl mx-auto">
            A four-layer architecture informed by medical device software patterns,
            enabling independent testing and component replacement.
          </p>
        </motion.div>

        <div className="space-y-3">
          {layers.map((layer, i) => (
            <motion.div
              key={layer.num}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="glass-card p-6 hover:border-white/10 transition-all group cursor-default"
            >
              <div className="flex flex-col md:flex-row md:items-center gap-6">
                <div className="flex items-center gap-4 md:w-64">
                  <span className="font-mono text-xs text-white/30 bg-white/5 px-2 py-1 rounded">
                    L{layer.num}
                  </span>
                  <layer.icon className={`w-5 h-5 text-${layer.color}`} />
                  <h3 className="text-lg font-medium">{layer.title}</h3>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-white/60 mb-2">{layer.desc}</p>
                  <span className="font-mono text-xs text-white/30">{layer.tech}</span>
                </div>
                <div className="flex gap-8">
                  {layer.metrics.map(m => (
                    <div key={m.label} className="text-right">
                      <div className="font-mono text-lg">{m.value}</div>
                      <div className="text-xs text-white/40">{m.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="flex flex-col items-center mt-8 text-white/20"
        >
          <div className="w-0.5 h-12 bg-gradient-to-t from-nc-accent to-transparent" />
          <ArrowRight className="w-4 h-4 rotate-90 -mt-1" />
          <span className="font-mono text-xs mt-2 uppercase tracking-widest">Data Flow</span>
        </motion.div>
      </div>
    </section>
  )
}

// Data Story Section
function DataStory() {
  const [selectedBand, setSelectedBand] = useState('alpha')

  const bands = [
    { id: 'delta', label: 'Delta', range: '0.5-4 Hz', amp: '20-200 µV' },
    { id: 'theta', label: 'Theta', range: '4-8 Hz', amp: '10-100 µV' },
    { id: 'alpha', label: 'Alpha', range: '8-13 Hz', amp: '15-50 µV' },
    { id: 'beta', label: 'Beta', range: '13-30 Hz', amp: '5-30 µV' },
    { id: 'gamma', label: 'Gamma', range: '30-50 Hz', amp: '<10 µV' },
  ]

  const channels = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']

  return (
    <section id="data" className="py-32 px-6 bg-nc-bg-secondary">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="font-mono text-nc-accent text-sm">02</span>
          <h2 className="section-title mt-4 mb-4">The Data Story</h2>
          <p className="text-white/60 max-w-xl mx-auto">
            Synthetic signals grounded in peer-reviewed literature, providing
            scientifically valid data for development and testing.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* EEG Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Brain className="w-5 h-5 text-nc-eeg" />
                <h3 className="text-lg font-medium">EEG Signals</h3>
              </div>
              <span className="font-mono text-xs text-nc-eeg bg-nc-eeg/10 px-2 py-1 rounded">
                8 Channels
              </span>
            </div>

            {/* Band Selector */}
            <div className="flex gap-2 mb-6 flex-wrap">
              {bands.map(band => (
                <button
                  key={band.id}
                  onClick={() => setSelectedBand(band.id)}
                  className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                    selectedBand === band.id
                      ? 'bg-nc-accent border-nc-accent text-white'
                      : 'border-white/10 text-white/60 hover:border-white/20'
                  }`}
                >
                  {band.label}
                </button>
              ))}
            </div>

            {/* Channel Grid */}
            <div className="grid grid-cols-4 gap-2 mb-6">
              {channels.map(ch => (
                <div
                  key={ch}
                  className="text-center py-2 bg-white/5 rounded text-xs font-mono text-white/60"
                >
                  {ch}
                </div>
              ))}
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-white/40">
                <span>Selected Band</span>
                <span className="font-mono">{bands.find(b => b.id === selectedBand)?.range}</span>
              </div>
              <div className="flex justify-between text-white/40">
                <span>Amplitude</span>
                <span className="font-mono">{bands.find(b => b.id === selectedBand)?.amp}</span>
              </div>
            </div>
          </motion.div>

          {/* ECG Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <Heart className="w-5 h-5 text-nc-ecg" />
                <h3 className="text-lg font-medium">ECG Signals</h3>
              </div>
              <span className="font-mono text-xs text-nc-ecg bg-nc-ecg/10 px-2 py-1 rounded">
                3 Leads
              </span>
            </div>

            {/* Lead Grid */}
            <div className="grid grid-cols-3 gap-2 mb-6">
              {['Lead I', 'Lead II', 'Lead III'].map(lead => (
                <div
                  key={lead}
                  className="text-center py-3 bg-white/5 rounded text-sm font-mono text-white/60"
                >
                  {lead}
                </div>
              ))}
            </div>

            {/* PQRST Info */}
            <div className="bg-black/30 rounded-lg p-4 mb-6">
              <div className="font-mono text-xs text-white/30 mb-2">PQRST MORPHOLOGY</div>
              <div className="flex justify-between text-sm">
                {['P', 'Q', 'R', 'S', 'T'].map(wave => (
                  <div key={wave} className="text-center">
                    <div className="text-nc-ecg font-medium">{wave}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-white/40">
                <span>Sample Rate</span>
                <span className="font-mono">250 Hz</span>
              </div>
              <div className="flex justify-between text-white/40">
                <span>HRV Model</span>
                <span className="font-mono">Task Force 1996</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Simulation Notice */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mt-8 p-4 border border-nc-risk-medium/20 bg-nc-risk-medium/5 rounded-lg flex gap-4"
        >
          <Shield className="w-5 h-5 text-nc-risk-medium flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-medium text-nc-risk-medium mb-1">Simulation Scope</h4>
            <p className="text-sm text-white/60">
              All physiological data is computationally generated. The system demonstrates
              architecture and methodology, not clinical capability. See DATA_BIBLE.md for specifications.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Decision Engine Section
function DecisionEngine() {
  const features = [
    { category: 'EEG Band Power', count: 40, icon: Brain, color: 'nc-eeg' },
    { category: 'EEG Statistics', count: 16, icon: BarChart3, color: 'nc-eeg' },
    { category: 'EEG Coherence', count: 8, icon: GitBranch, color: 'nc-eeg' },
    { category: 'HRV Metrics', count: 12, icon: Heart, color: 'nc-ecg' },
  ]

  return (
    <section id="intelligence" className="py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="font-mono text-nc-accent text-sm">03</span>
          <h2 className="section-title mt-4 mb-4">The Decision Engine</h2>
          <p className="text-white/60 max-w-xl mx-auto">
            An interpretable ML ensemble that prioritizes explainability over
            raw accuracy, providing transparent risk assessments.
          </p>
        </motion.div>

        {/* Feature Categories */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
          {features.map((f, i) => (
            <motion.div
              key={f.category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="glass-card p-5"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-2 h-2 rounded-full bg-${f.color}`} />
                <span className="text-sm text-white/60">{f.category}</span>
              </div>
              <div className="font-mono text-3xl text-nc-accent">{f.count}</div>
              <div className="text-xs text-white/40 mt-1">features</div>
            </motion.div>
          ))}
        </div>

        {/* Ensemble Diagram */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-8"
        >
          <h3 className="text-lg font-medium text-center mb-8">Ensemble Architecture</h3>

          <div className="flex flex-col md:flex-row items-center justify-center gap-8">
            {/* XGBoost */}
            <div className="glass-card p-5 w-64 border-l-2 border-nc-hrv">
              <div className="flex justify-between items-center mb-3">
                <span className="font-medium">XGBoost</span>
                <span className="font-mono text-sm text-nc-accent bg-nc-accent/10 px-2 py-0.5 rounded">
                  60%
                </span>
              </div>
              <p className="text-xs text-white/40 mb-4">
                Gradient boosted trees for feature-based classification
              </p>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-nc-risk-low/10 text-nc-risk-low py-1 rounded">15%</div>
                <div className="bg-nc-risk-medium/10 text-nc-risk-medium py-1 rounded">55%</div>
                <div className="bg-nc-risk-high/10 text-nc-risk-high py-1 rounded">30%</div>
              </div>
            </div>

            {/* Combiner */}
            <div className="flex flex-col items-center gap-2">
              <div className="w-10 h-10 rounded-full border border-nc-accent flex items-center justify-center text-nc-accent">
                +
              </div>
              <span className="font-mono text-xs text-white/40">weighted</span>
            </div>

            {/* BiLSTM */}
            <div className="glass-card p-5 w-64 border-l-2 border-nc-eeg">
              <div className="flex justify-between items-center mb-3">
                <span className="font-medium">BiLSTM</span>
                <span className="font-mono text-sm text-nc-accent bg-nc-accent/10 px-2 py-0.5 rounded">
                  40%
                </span>
              </div>
              <p className="text-xs text-white/40 mb-4">
                Bidirectional LSTM for temporal pattern recognition
              </p>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-nc-risk-low/10 text-nc-risk-low py-1 rounded">12%</div>
                <div className="bg-nc-risk-medium/10 text-nc-risk-medium py-1 rounded">60%</div>
                <div className="bg-nc-risk-high/10 text-nc-risk-high py-1 rounded">28%</div>
              </div>
            </div>
          </div>

          {/* Result */}
          <div className="flex justify-center mt-8">
            <div className="glass-card p-5 w-80">
              <div className="flex justify-between items-center mb-4">
                <span className="text-sm font-medium">Ensemble Output</span>
                <span className="text-xs text-nc-risk-medium bg-nc-risk-medium/10 px-2 py-0.5 rounded">
                  MEDIUM RISK
                </span>
              </div>
              <div className="h-2 bg-gradient-to-r from-nc-risk-low via-nc-risk-medium to-nc-risk-high rounded-full mb-2 relative">
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-white rounded shadow-lg"
                  style={{ left: '57%' }}
                />
              </div>
              <div className="flex justify-between text-xs text-white/40">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Explainability */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-8 glass-card p-6"
        >
          <h3 className="text-lg font-medium mb-6">Interpretability</h3>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h4 className="text-sm text-white/60 mb-4">Top Contributing Features</h4>
              {[
                { name: 'rmssd', value: '28.5 ms', importance: 0.23 },
                { name: 'O1_alpha_power', value: '142 µV²', importance: 0.18 },
                { name: 'lf_hf_ratio', value: '1.85', importance: 0.14 },
                { name: 'sdnn', value: '52 ms', importance: 0.11 },
              ].map(f => (
                <div key={f.name} className="mb-4">
                  <div
                    className="h-1 bg-nc-accent rounded mb-2"
                    style={{ width: `${f.importance * 400}%`, opacity: 0.6 }}
                  />
                  <div className="flex justify-between text-sm">
                    <span className="font-mono text-white/80">{f.name}</span>
                    <span className="text-white/40">{f.value}</span>
                  </div>
                </div>
              ))}
            </div>
            <div>
              <h4 className="text-sm text-white/60 mb-4">Interpretation Notes</h4>
              <ul className="space-y-3 text-sm text-white/60">
                <li className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-nc-accent mt-2" />
                  Reduced HRV metrics suggest elevated sympathetic activation
                </li>
                <li className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-nc-accent mt-2" />
                  Elevated alpha asymmetry consistent with stress response patterns
                </li>
                <li className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-nc-accent mt-2" />
                  LF/HF ratio indicates parasympathetic withdrawal
                </li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Device Connection Section
function DeviceConnection() {
  const adapters = [
    {
      type: 'simulated',
      title: 'Simulated',
      desc: 'Synthetic signals for development',
      status: 'Implemented',
      statusColor: 'nc-risk-low',
    },
    {
      type: 'ble',
      title: 'BLE',
      desc: 'Bluetooth Low Energy wearables',
      status: 'Interface Ready',
      statusColor: 'nc-risk-medium',
    },
    {
      type: 'serial',
      title: 'Serial',
      desc: 'UART/USB microcontrollers',
      status: 'Implemented',
      statusColor: 'nc-risk-low',
    },
  ]

  return (
    <section id="device" className="py-32 px-6 bg-nc-bg-secondary">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="font-mono text-nc-accent text-sm">04</span>
          <h2 className="section-title mt-4 mb-4">Connect Your Device</h2>
          <p className="text-white/60 max-w-xl mx-auto">
            The pluggable adapter architecture allows seamless switching between
            simulated and real sensor hardware.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {adapters.map((adapter, i) => (
            <motion.div
              key={adapter.type}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="glass-card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-medium">{adapter.title}</h3>
                <span className={`text-xs text-${adapter.statusColor} bg-${adapter.statusColor}/10 px-2 py-0.5 rounded`}>
                  {adapter.status}
                </span>
              </div>
              <p className="text-sm text-white/60 mb-4">{adapter.desc}</p>
              <code className="block text-xs font-mono text-white/40 bg-black/30 p-3 rounded">
                get_adapter('{adapter.type}')
              </code>
            </motion.div>
          ))}
        </div>

        {/* Quick Start */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-medium mb-6">Quick Start</h3>
          <div className="bg-black/30 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm font-mono text-white/80">
{`from cloud.device_adapters import get_adapter

# Connect to simulated device
adapter = get_adapter('simulated', seed=42)
adapter.connect()
adapter.start_stream()

# Read physiological packets
packet = adapter.read_packet()
print(f"EEG: {len(packet.eeg)} channels")
print(f"ECG: {len(packet.ecg)} leads")

adapter.stop()`}
            </pre>
          </div>
          <p className="text-sm text-white/40 mt-4">
            See <code className="text-nc-accent">docs/DEVICE_INTEGRATION.md</code> for complete setup instructions.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

// Verification Section
function Verification() {
  const checks = [
    { category: 'Directory Structure', count: 14, icon: Layers },
    { category: 'Source Files', count: 12, icon: FileText },
    { category: 'Documentation', count: 5, icon: FileText },
    { category: 'ML Pipeline', count: 8, icon: Cpu },
    { category: 'Device Adapters', count: 6, icon: Settings },
    { category: 'Code Quality', count: 1, icon: Shield },
  ]

  return (
    <section id="verification" className="py-32 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <span className="font-mono text-nc-accent text-sm">05</span>
          <h2 className="section-title mt-4 mb-4">Verification & Trust</h2>
          <p className="text-white/60 max-w-xl mx-auto">
            Automated verification ensures reproducibility and system integrity
            across all components.
          </p>
        </motion.div>

        {/* Summary Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-8 mb-8 flex flex-col md:flex-row items-center justify-center gap-12"
        >
          <div className="text-center">
            <div className="font-mono text-5xl text-nc-risk-low">67</div>
            <div className="text-sm text-white/60 mt-2">Total Checks</div>
          </div>
          <div className="w-px h-12 bg-white/10 hidden md:block" />
          <div className="text-center">
            <div className="font-mono text-5xl text-nc-risk-low">100%</div>
            <div className="text-sm text-white/60 mt-2">Pass Rate</div>
          </div>
          <div className="w-px h-12 bg-white/10 hidden md:block" />
          <div className="text-center">
            <div className="font-mono text-5xl text-nc-risk-low">42</div>
            <div className="text-sm text-white/60 mt-2">Fixed Seed</div>
          </div>
        </motion.div>

        {/* Check Categories */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {checks.map((check, i) => (
            <motion.div
              key={check.category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="glass-card p-4 flex items-center gap-4"
            >
              <div className="w-10 h-10 rounded-lg bg-nc-risk-low/10 flex items-center justify-center">
                <Check className="w-5 h-5 text-nc-risk-low" />
              </div>
              <div className="flex-1">
                <div className="text-sm font-medium">{check.category}</div>
                <div className="font-mono text-xs text-nc-risk-low">{check.count} passed</div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Run Command */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-medium mb-4">Run Verification</h3>
          <div className="bg-black/30 rounded-lg p-4 font-mono text-sm">
            <span className="text-white/40">$</span> python verify_system.py
          </div>
          <p className="text-sm text-white/40 mt-4">
            Verifies all components, dependencies, and outputs detailed results.
          </p>
        </motion.div>
      </div>
    </section>
  )
}

// Credits Section
function Credits() {
  return (
    <section id="credits" className="py-32 px-6 bg-nc-bg-secondary">
      <div className="max-w-3xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl font-light mb-2">NeuroCardiac Shield</h2>
          <p className="text-sm text-white/40 mb-12">
            Integrated Brain-Heart Monitoring System
          </p>

          {/* Authors */}
          <div className="flex justify-center gap-16 mb-12">
            <div>
              <div className="font-medium mb-1">Mohd Sarfaraz Faiyaz</div>
              <div className="text-xs text-white/40">Author</div>
            </div>
            <div>
              <div className="font-medium mb-1">Vaibhav D. Chandgir</div>
              <div className="text-xs text-white/40">Author</div>
            </div>
          </div>

          {/* Institution */}
          <div className="mb-8">
            <div className="text-sm text-white/60">NYU Tandon School of Engineering</div>
            <div className="text-xs text-white/40">Department of Electrical and Computer Engineering</div>
          </div>

          {/* Advisor */}
          <div className="inline-block glass-card px-6 py-4 mb-8">
            <div className="text-xs text-white/40 mb-1">Advisor</div>
            <div className="font-medium">Dr. Matthew Campisi</div>
          </div>

          {/* Term */}
          <div className="font-mono text-nc-accent mb-12">
            ECE-GY 9953 · Fall 2025
          </div>

          {/* Links */}
          <div className="flex justify-center gap-4 mb-12">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 border border-white/10 rounded-lg text-sm text-white/60 hover:border-nc-accent/50 hover:text-white transition-all"
            >
              <ExternalLink className="w-4 h-4" />
              View Repository
            </a>
          </div>

          {/* Disclaimer */}
          <div className="text-xs text-white/30 leading-relaxed max-w-xl mx-auto border-t border-white/10 pt-8">
            This is an academic demonstration system developed for NYU Tandon's
            Advanced Project course. All physiological data is computationally generated.
            The system is not FDA-cleared, clinically validated, or suitable for
            medical diagnosis or patient care.
          </div>
        </motion.div>
      </div>
    </section>
  )
}

// Main Page
export default function Home() {
  return (
    <main>
      <Navigation />
      <Hero />
      <Architecture />
      <DataStory />
      <DecisionEngine />
      <DeviceConnection />
      <Verification />
      <Credits />
    </main>
  )
}
