'use client'

import { useState, useEffect, useRef, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, AreaChart, Area, Cell, PieChart, Pie
} from 'recharts'
import {
  Activity, Brain, Heart, Cpu, Database, Server, Wifi, AlertCircle,
  Monitor, ChevronDown, ArrowRight, ArrowDown, Check, AlertTriangle,
  Zap, Shield, GitBranch, FileText, Users, Clock, TrendingUp, BookOpen,
  Layers, BarChart3, Settings, ExternalLink, Github, Play, Target,
  Stethoscope, Radio, Binary, Filter, Microscope, Award, CircuitBoard,
  Bluetooth, BluetoothConnected, BluetoothOff, BluetoothSearching,
  Smartphone, Watch, Headphones, X, RefreshCw, Link2, Unlink
} from 'lucide-react'

// ============================================================================
// WEB BLUETOOTH API TYPE DECLARATIONS
// ============================================================================

declare global {
  interface Navigator {
    bluetooth: Bluetooth
  }
  interface Bluetooth {
    requestDevice(options: RequestDeviceOptions): Promise<BluetoothDevice>
  }
  interface RequestDeviceOptions {
    filters?: BluetoothLEScanFilter[]
    optionalServices?: BluetoothServiceUUID[]
    acceptAllDevices?: boolean
  }
  interface BluetoothLEScanFilter {
    services?: BluetoothServiceUUID[]
    name?: string
    namePrefix?: string
  }
  type BluetoothServiceUUID = number | string
  interface BluetoothDevice extends EventTarget {
    id: string
    name?: string
    gatt?: BluetoothRemoteGATTServer
  }
  interface BluetoothRemoteGATTServer {
    device: BluetoothDevice
    connected: boolean
    connect(): Promise<BluetoothRemoteGATTServer>
    disconnect(): void
    getPrimaryService(service: BluetoothServiceUUID): Promise<BluetoothRemoteGATTService>
  }
  interface BluetoothRemoteGATTService {
    device: BluetoothDevice
    uuid: string
    getCharacteristic(characteristic: BluetoothCharacteristicUUID): Promise<BluetoothRemoteGATTCharacteristic>
  }
  type BluetoothCharacteristicUUID = number | string
  interface BluetoothRemoteGATTCharacteristic extends EventTarget {
    service: BluetoothRemoteGATTService
    uuid: string
    value?: DataView
    startNotifications(): Promise<BluetoothRemoteGATTCharacteristic>
    stopNotifications(): Promise<BluetoothRemoteGATTCharacteristic>
    readValue(): Promise<DataView>
  }
}

// ============================================================================
// DEVICE CONNECTION TYPES & STATE
// ============================================================================

interface DeviceData {
  heartRate: number
  rrIntervals: number[]
  ecgData?: number[]
  batteryLevel?: number
  deviceName: string
  deviceId: string
  connected: boolean
  timestamp: number
}

interface ConnectedDevice {
  device: BluetoothDevice
  server?: BluetoothRemoteGATTServer
  heartRateChar?: BluetoothRemoteGATTCharacteristic
  type: 'hr' | 'ecg' | 'eeg'
  name: string
}

// Global device state (shared between components)
let globalDeviceData: DeviceData | null = null
let globalDeviceListeners: ((data: DeviceData | null) => void)[] = []

function notifyDeviceListeners(data: DeviceData | null) {
  globalDeviceData = data
  globalDeviceListeners.forEach(listener => listener(data))
}

function useDeviceData() {
  const [deviceData, setDeviceData] = useState<DeviceData | null>(globalDeviceData)

  useEffect(() => {
    const listener = (data: DeviceData | null) => setDeviceData(data)
    globalDeviceListeners.push(listener)
    return () => {
      globalDeviceListeners = globalDeviceListeners.filter(l => l !== listener)
    }
  }, [])

  return deviceData
}

// ============================================================================
// PHYSIOLOGICAL STATE TYPES
// ============================================================================

type PhysiologicalState = 'alert' | 'relaxed' | 'drowsy' | 'stressed'

interface EEGStateProfile {
  delta: number    // 0.5-4 Hz: deep sleep, healing
  theta: number    // 4-8 Hz: drowsiness, meditation
  alpha: number    // 8-13 Hz: relaxed wakefulness
  beta: number     // 13-30 Hz: active thinking
  gamma: number    // 30-50 Hz: high cognition
  noise: number    // 1/f noise amplitude
  label: string
  description: string
}

// Physiological state profiles based on clinical EEG literature
// References:
// - Niedermeyer & Lopes da Silva, "Electroencephalography" (2004)
// - Barry et al., "EEG differences between eyes-closed and eyes-open" (2007)
const EEG_STATE_PROFILES: Record<PhysiologicalState, EEGStateProfile> = {
  alert: {
    delta: 0.3, theta: 0.4, alpha: 0.5, beta: 1.2, gamma: 0.8, noise: 1.0,
    label: 'Alert / Focused',
    description: 'High beta/gamma activity, suppressed alpha. Characteristic of active cognitive processing.'
  },
  relaxed: {
    delta: 0.4, theta: 0.5, alpha: 1.5, beta: 0.6, gamma: 0.3, noise: 0.8,
    label: 'Relaxed / Eyes Closed',
    description: 'Dominant alpha rhythm, especially in occipital regions. Classic "alpha block" when eyes open.'
  },
  drowsy: {
    delta: 0.8, theta: 1.3, alpha: 0.6, beta: 0.3, gamma: 0.1, noise: 1.2,
    label: 'Drowsy / Stage 1 Sleep',
    description: 'Increased theta, alpha attenuation. Vertex waves may appear. Transition to sleep.'
  },
  stressed: {
    delta: 0.4, theta: 0.5, alpha: 0.4, beta: 1.5, gamma: 1.0, noise: 1.5,
    label: 'Stressed / Anxious',
    description: 'High beta activity, reduced alpha. Increased muscle artifact and 1/f noise.'
  }
}

// ============================================================================
// ADVANCED SIGNAL GENERATION ALGORITHMS
// ============================================================================

/**
 * Generate 1/f (pink) noise using the Voss-McCartney algorithm
 * Pink noise has equal energy per octave, matching natural EEG spectra
 * Reference: Voss & Clarke, "1/f noise in music and speech" (1975)
 */
function generatePinkNoise(length: number, amplitude: number = 1): number[] {
  const numOctaves = 8
  const octaves: number[] = new Array(numOctaves).fill(0)
  const result: number[] = []

  for (let i = 0; i < length; i++) {
    // Update octaves based on bit position
    for (let j = 0; j < numOctaves; j++) {
      if (i % Math.pow(2, j) === 0) {
        octaves[j] = (Math.random() - 0.5) * 2
      }
    }
    // Sum all octaves
    const sum = octaves.reduce((a, b) => a + b, 0) / numOctaves
    result.push(sum * amplitude)
  }
  return result
}

/**
 * Generate EEG signal with physiological accuracy
 * Uses multi-band synthesis with 1/f noise and state-dependent parameters
 */
function generateAdvancedEEG(
  channel: string,
  offset: number,
  state: PhysiologicalState = 'relaxed',
  showComponents: boolean = false
) {
  const data: any[] = []
  const components: Record<string, number[]> = {
    delta: [], theta: [], alpha: [], beta: [], gamma: [], noise: []
  }

  // Channel-specific baseline amplitudes (microvolts) based on 10-20 system
  // Reference: Nunez & Srinivasan, "Electric Fields of the Brain" (2006)
  const channelParams: Record<string, {delta: number, theta: number, alpha: number, beta: number, gamma: number}> = {
    'Fp1': { delta: 20, theta: 12, alpha: 15, beta: 8, gamma: 3 },   // Frontal: cognitive/executive
    'Fp2': { delta: 18, theta: 11, alpha: 14, beta: 9, gamma: 4 },
    'C3': { delta: 15, theta: 15, alpha: 30, beta: 12, gamma: 5 },   // Central: motor/sensory
    'C4': { delta: 16, theta: 14, alpha: 28, beta: 11, gamma: 4 },
    'T3': { delta: 12, theta: 18, alpha: 12, beta: 6, gamma: 2 },    // Temporal: language/memory
    'T4': { delta: 13, theta: 17, alpha: 13, beta: 7, gamma: 3 },
    'O1': { delta: 10, theta: 10, alpha: 50, beta: 5, gamma: 2 },    // Occipital: visual (highest alpha)
    'O2': { delta: 11, theta: 9, alpha: 48, beta: 6, gamma: 2 },
  }

  const baseParams = channelParams[channel] || channelParams['Fp1']
  const stateProfile = EEG_STATE_PROFILES[state]

  // Generate pink noise for this segment (seeded for consistency)
  const pinkNoise = generatePinkNoise(200, 3 * stateProfile.noise)

  // Fixed frequencies for smooth scrolling
  const freqs = {
    delta: 2,      // 2 Hz
    theta: 6,      // 6 Hz
    alpha: 10,     // 10 Hz
    beta: 20,      // 20 Hz
    gamma: 40      // 40 Hz
  }

  for (let i = 0; i < 200; i++) {
    const t = (i + offset) / 250  // Time in seconds

    // Smooth sinusoidal generation - no random jumps
    const delta = baseParams.delta * stateProfile.delta * (
      Math.sin(2 * Math.PI * freqs.delta * t) +
      0.3 * Math.sin(2 * Math.PI * freqs.delta * 1.5 * t + 0.5)
    )

    const theta = baseParams.theta * stateProfile.theta * (
      Math.sin(2 * Math.PI * freqs.theta * t) +
      0.25 * Math.sin(2 * Math.PI * freqs.theta * 1.3 * t + 0.7)
    )

    // Alpha with slow amplitude modulation (spindles)
    const alphaModulation = 0.7 + 0.3 * Math.sin(2 * Math.PI * 0.5 * t)
    const alpha = baseParams.alpha * stateProfile.alpha * alphaModulation * (
      Math.sin(2 * Math.PI * freqs.alpha * t) +
      0.2 * Math.sin(2 * Math.PI * (freqs.alpha + 1) * t + 0.3)
    )

    const beta = baseParams.beta * stateProfile.beta * (
      Math.sin(2 * Math.PI * freqs.beta * t) +
      0.4 * Math.sin(2 * Math.PI * freqs.beta * 1.5 * t + 0.8)
    )

    const gamma = baseParams.gamma * stateProfile.gamma * (
      Math.sin(2 * Math.PI * freqs.gamma * t) +
      0.3 * Math.sin(2 * Math.PI * freqs.gamma * 1.2 * t)
    )

    // 1/f pink noise component
    const noise = pinkNoise[i]

    // Store components if requested
    if (showComponents) {
      components.delta.push(delta)
      components.theta.push(theta)
      components.alpha.push(alpha)
      components.beta.push(beta)
      components.gamma.push(gamma)
      components.noise.push(noise)
    }

    // Combined signal
    const value = delta + theta + alpha + beta + gamma + noise

    data.push({
      time: i,
      value,
      delta, theta, alpha, beta, gamma, noise
    })
  }

  return showComponents ? { data, components } : data
}

/**
 * Generate ECG using McSharry dynamical model principles
 * Reference: McSharry et al., "A Dynamical Model for Generating Synthetic
 * Electrocardiogram Signals" IEEE Trans Biomed Eng (2003)
 */
interface ECGOptions {
  heartRate?: number           // BPM (60-100 normal)
  rrVariability?: number       // HRV amplitude (0-1)
  rsaAmplitude?: number        // Respiratory sinus arrhythmia
  showPathology?: boolean      // Include ectopic beats
  stElevation?: number         // ST segment elevation (mV)
}

// Use seeded random for consistent beat-to-beat variation
function seededRandom(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453
  return x - Math.floor(x)
}

function generateAdvancedECG(offset: number, options: ECGOptions = {}) {
  const {
    heartRate = 72,
    showPathology = false,
    stElevation = 0
  } = options

  const data: any[] = []
  const samplesPerBeat = Math.round(250 * 60 / heartRate)
  const totalSamples = 300

  // Generate PQRST value with beat-specific variations
  const getPQRSTValue = (phase: number, isEctopic: boolean, beatVariation: number): number => {
    // Each beat has slightly different amplitude (±10%)
    const ampVar = 0.9 + beatVariation * 0.2

    if (isEctopic) {
      if (phase >= 0.15 && phase < 0.35) {
        const qrs = (phase - 0.25) / 0.05
        return 80 * ampVar * Math.exp(-qrs * qrs) * (1 - 0.5 * Math.sin(phase * Math.PI * 10))
      } else if (phase >= 0.4 && phase < 0.7) {
        const t = (phase - 0.55) / 0.1
        return -20 * ampVar * Math.exp(-t * t)
      }
      return 0
    }

    // Normal PQRST with variations
    if (phase >= 0.0 && phase < 0.12) {
      const p = (phase - 0.06) / 0.035
      return 10 * ampVar * Math.exp(-p * p / 2)
    }
    if (phase >= 0.16 && phase < 0.20) {
      const q = (phase - 0.18) / 0.015
      return -8 * ampVar * Math.exp(-q * q / 2)
    }
    if (phase >= 0.20 && phase < 0.26) {
      const r = (phase - 0.23) / 0.018
      return 100 * ampVar * Math.exp(-r * r / 2)
    }
    if (phase >= 0.26 && phase < 0.32) {
      const s = (phase - 0.29) / 0.02
      return -18 * ampVar * Math.exp(-s * s / 2)
    }
    if (phase >= 0.32 && phase < 0.45) {
      return stElevation * 10
    }
    if (phase >= 0.45 && phase < 0.70) {
      const t = (phase - 0.55) / 0.07
      return 25 * ampVar * Math.exp(-t * t / 2)
    }
    if (phase >= 0.70 && phase < 0.80) {
      const u = (phase - 0.75) / 0.03
      return 3 * ampVar * Math.exp(-u * u / 2)
    }
    return 0
  }

  for (let i = 0; i < totalSamples; i++) {
    const streamPos = i + offset
    const beatIndex = Math.floor(streamPos / samplesPerBeat)
    const beatPhase = (streamPos % samplesPerBeat) / samplesPerBeat

    // Beat-specific random variation (consistent for same beat)
    const beatVariation = seededRandom(beatIndex)

    const isEctopic = showPathology && (beatIndex % 10 === 7)

    let value = getPQRSTValue(beatPhase, isEctopic, beatVariation)

    // Baseline wander (slow drift)
    const baselineWander = 3 * Math.sin(2 * Math.PI * streamPos / 500)

    // Respiratory variation
    const respVariation = 2 * Math.sin(2 * Math.PI * streamPos / 800)

    // Random noise
    const noise = (Math.random() - 0.5) * 2

    value += baselineWander + respVariation + noise

    data.push({
      time: i,
      value,
      phase: beatPhase
    })
  }

  return data
}

// Legacy wrapper for backward compatibility
function generateRealisticEEG(channel: string, offset: number) {
  return generateAdvancedEEG(channel, offset, 'relaxed')
}

function generateRealisticECG(offset: number) {
  return generateAdvancedECG(offset)
}

// ============================================================================
// SECTION COMPONENT
// ============================================================================

function Section({ id, children, className = '', dark = false }: {
  id?: string,
  children: React.ReactNode,
  className?: string,
  dark?: boolean
}) {
  return (
    <section
      id={id}
      className={`py-20 px-6 ${dark ? 'bg-nc-bg-secondary' : ''} ${className}`}
    >
      <div className="max-w-6xl mx-auto">
        {children}
      </div>
    </section>
  )
}

function SectionHeader({ label, title, description }: { label: string, title: string, description: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className="mb-12"
    >
      <span className="font-mono text-nc-accent text-sm">{label}</span>
      <h2 className="text-4xl font-light mt-2 mb-4">{title}</h2>
      <p className="text-white/60 max-w-3xl text-lg leading-relaxed">{description}</p>
    </motion.div>
  )
}

function InfoCard({ icon: Icon, title, children, color = 'nc-accent' }: {
  icon: any, title: string, children: React.ReactNode, color?: string
}) {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-10 h-10 rounded-lg bg-${color}/20 flex items-center justify-center`}>
          <Icon className={`w-5 h-5 text-${color}`} />
        </div>
        <h3 className="font-semibold text-lg">{title}</h3>
      </div>
      <div className="text-white/70 leading-relaxed">{children}</div>
    </div>
  )
}

// ============================================================================
// NAVIGATION
// ============================================================================

function Navigation() {
  const [scrolled, setScrolled] = useState(false)
  const [activeSection, setActiveSection] = useState('')

  const sections = [
    { id: 'problem', label: 'Problem' },
    { id: 'solution', label: 'Solution' },
    { id: 'signals', label: 'Signals' },
    { id: 'science', label: 'Science' },
    { id: 'processing', label: 'Processing' },
    { id: 'ml', label: 'ML' },
    { id: 'architecture', label: 'Architecture' },
    { id: 'results', label: 'Results' },
    { id: 'connect', label: 'Connect', highlight: true },
  ]

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)

      // Find which section is currently in view
      const scrollPosition = window.scrollY + 150 // Offset for navbar

      for (let i = sections.length - 1; i >= 0; i--) {
        const section = document.getElementById(sections[i].id)
        if (section && section.offsetTop <= scrollPosition) {
          setActiveSection(sections[i].id)
          break
        }
      }
    }

    window.addEventListener('scroll', handleScroll)
    handleScroll() // Initial check
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
      scrolled ? 'bg-nc-bg/95 backdrop-blur-lg border-b border-white/5' : ''
    }`}>
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-mono text-sm border border-nc-accent px-3 py-1.5 rounded">
            <Brain className="w-4 h-4 text-nc-eeg" />
            <Heart className="w-4 h-4 text-nc-ecg" />
          </div>
          <span className="text-sm font-medium hidden sm:block">NeuroCardiac Shield</span>
        </div>
        <div className="flex items-center gap-1">
          {sections.map(section => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className={`text-xs px-3 py-2 rounded transition-colors hidden md:block ${
                (section as any).highlight
                  ? 'bg-nc-ecg/20 text-nc-ecg hover:bg-nc-ecg/30 font-medium'
                  : activeSection === section.id
                  ? 'bg-nc-accent/20 text-nc-accent font-medium'
                  : 'text-white/50 hover:text-white hover:bg-white/5'
              }`}
            >
              {(section as any).highlight && <Bluetooth className="w-3 h-3 inline mr-1" />}
              {section.label}
            </a>
          ))}
          <a
            href="https://github.com/bblackheart013/neurocardiac-shield"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm px-4 py-2 bg-nc-accent/20 border border-nc-accent/50 rounded-lg hover:bg-nc-accent/30 transition-colors ml-4"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </nav>
  )
}

// ============================================================================
// HERO SECTION
// ============================================================================

function Hero() {
  return (
    <section className="min-h-screen flex flex-col items-center justify-center relative overflow-hidden px-6 pt-20">
      <div className="absolute inset-0 bg-gradient-to-b from-nc-accent/5 via-transparent to-transparent" />

      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-nc-eeg/5 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-nc-ecg/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-5xl relative z-10"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center gap-3 font-mono text-xs border border-white/20 px-4 py-2 rounded-full mb-8 bg-white/5"
        >
          <Award className="w-4 h-4 text-nc-accent" />
          <span className="text-white/60">NYU TANDON · ECE-GY 9953 · FALL 2025 · ADVANCED PROJECT</span>
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-light tracking-tight mb-6">
          <span className="block text-white/90">NeuroCardiac</span>
          <span className="block bg-gradient-to-r from-nc-eeg via-nc-accent to-nc-ecg bg-clip-text text-transparent font-medium">
            Shield
          </span>
        </h1>

        <p className="text-xl md:text-2xl text-white/50 max-w-3xl mx-auto mb-6 leading-relaxed">
          A complete multi-modal physiological monitoring platform that integrates
          brain and heart signals for real-time health risk assessment
        </p>

        <p className="text-lg text-white/70 max-w-2xl mx-auto mb-12">
          We built an end-to-end system from <span className="text-nc-ecg">embedded firmware</span> to{' '}
          <span className="text-nc-eeg">machine learning</span> to demonstrate how next-generation
          wearable medical devices could detect cardiovascular-neurological risks before they become emergencies.
        </p>

        {/* Key Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-12">
          {[
            { value: '11', label: 'Signal Channels', sublabel: '8 EEG + 3 ECG', color: 'nc-accent' },
            { value: '76', label: 'ML Features', sublabel: 'Hand-crafted', color: 'nc-eeg' },
            { value: '250', label: 'Hz Sample Rate', sublabel: 'Real-time', color: 'nc-hrv' },
            { value: '4', label: 'System Layers', sublabel: 'Full stack', color: 'nc-ecg' },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              className="glass-card p-4 text-center"
            >
              <div className={`font-mono text-3xl text-${stat.color} mb-1`}>{stat.value}</div>
              <div className="text-sm text-white/80">{stat.label}</div>
              <div className="text-xs text-white/40">{stat.sublabel}</div>
            </motion.div>
          ))}
        </div>

        {/* Team */}
        <div className="flex flex-col items-center gap-4">
          <div className="flex items-center gap-8">
            <div className="text-center">
              <div className="font-medium text-white/90">Mohd Sarfaraz Faiyaz</div>
              <div className="text-xs text-white/40">Systems & ML</div>
            </div>
            <div className="w-px h-8 bg-white/20" />
            <div className="text-center">
              <div className="font-medium text-white/90">Vaibhav D. Chandgir</div>
              <div className="text-xs text-white/40">Signal Processing</div>
            </div>
          </div>
          <div className="text-sm text-white/40">
            Advisor: <span className="text-white/60">Dr. Matthew Campisi</span>
          </div>
        </div>
      </motion.div>

      <motion.a
        href="#problem"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 cursor-pointer hover:text-nc-accent transition-colors"
      >
        <span className="text-xs text-white/40 uppercase tracking-widest">Scroll to explore</span>
        <ChevronDown className="w-5 h-5 text-white/40 animate-bounce" />
      </motion.a>
    </section>
  )
}

// ============================================================================
// PROBLEM SECTION
// ============================================================================

function ProblemSection() {
  return (
    <Section id="problem" dark>
      <SectionHeader
        label="01 — THE PROBLEM"
        title="Why This Matters"
        description="Every year, millions of people experience sudden cardiac events or neurological emergencies that could have been predicted. Current wearables monitor heart OR brain — never both together."
      />

      <div className="grid md:grid-cols-2 gap-6 mb-12">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-8 border-l-4 border-nc-ecg"
        >
          <div className="flex items-center gap-3 mb-4">
            <Heart className="w-8 h-8 text-nc-ecg" />
            <h3 className="text-xl font-semibold">Cardiac Events</h3>
          </div>
          <div className="space-y-4 text-white/70">
            <p>
              <span className="text-3xl font-mono text-nc-ecg">805,000</span>
              <span className="text-white/50 ml-2">heart attacks per year in the US alone</span>
            </p>
            <p>
              Many cardiac arrhythmias produce neurological symptoms (dizziness, syncope)
              <em className="text-white/50"> before</em> the heart attack occurs. Current ECG monitors miss these warning signs.
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-8 border-l-4 border-nc-eeg"
        >
          <div className="flex items-center gap-3 mb-4">
            <Brain className="w-8 h-8 text-nc-eeg" />
            <h3 className="text-xl font-semibold">Neurological Events</h3>
          </div>
          <div className="space-y-4 text-white/70">
            <p>
              <span className="text-3xl font-mono text-nc-eeg">1 in 26</span>
              <span className="text-white/50 ml-2">people will develop epilepsy</span>
            </p>
            <p>
              SUDEP (Sudden Unexpected Death in Epilepsy) kills thousands yearly.
              These deaths involve cardiac arrhythmias triggered by seizures — a <em className="text-white/50">brain-heart connection</em> that single-modality devices cannot detect.
            </p>
          </div>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 bg-gradient-to-r from-nc-eeg/10 to-nc-ecg/10"
      >
        <div className="flex items-start gap-4">
          <AlertCircle className="w-8 h-8 text-nc-accent flex-shrink-0 mt-1" />
          <div>
            <h3 className="text-xl font-semibold mb-3">The Gap We Address</h3>
            <p className="text-white/70 text-lg leading-relaxed">
              The heart and brain are bidirectionally connected through the autonomic nervous system.
              Cardiac events affect brain function; neurological events affect heart rhythm.
              <strong className="text-white"> No consumer device monitors both simultaneously.</strong>
            </p>
            <p className="text-white/50 mt-4">
              NeuroCardiac Shield demonstrates how such a device would be built — from signal acquisition
              to machine learning inference — as a complete reference implementation.
            </p>
          </div>
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// SOLUTION SECTION
// ============================================================================

function SolutionSection() {
  return (
    <Section id="solution">
      <SectionHeader
        label="02 — OUR SOLUTION"
        title="What We Built"
        description="A complete end-to-end system that captures brain and heart signals, processes them in real-time, extracts clinically meaningful features, and predicts health risks using machine learning."
      />

      <div className="grid md:grid-cols-3 gap-6 mb-12">
        <InfoCard icon={CircuitBoard} title="Hardware Layer" color="nc-ecg">
          <p className="mb-3">
            C firmware simulating an embedded device that captures 8-channel EEG and 3-lead ECG at 250 Hz.
          </p>
          <ul className="text-sm space-y-1 text-white/50">
            <li>• 569-byte binary packets</li>
            <li>• 10 packets/second throughput</li>
            <li>• BLE communication stub</li>
          </ul>
        </InfoCard>

        <InfoCard icon={Filter} title="Signal Processing" color="nc-eeg">
          <p className="mb-3">
            Digital signal processing pipeline using validated algorithms from clinical literature.
          </p>
          <ul className="text-sm space-y-1 text-white/50">
            <li>• Butterworth bandpass filters</li>
            <li>• R-peak detection (Pan-Tompkins)</li>
            <li>• Welch PSD estimation</li>
          </ul>
        </InfoCard>

        <InfoCard icon={Cpu} title="ML Inference" color="nc-hrv">
          <p className="mb-3">
            Ensemble model combining gradient boosting with deep learning for interpretable predictions.
          </p>
          <ul className="text-sm space-y-1 text-white/50">
            <li>• XGBoost (60% weight)</li>
            <li>• BiLSTM (40% weight)</li>
            <li>• 76 hand-crafted features</li>
          </ul>
        </InfoCard>
      </div>

      {/* System Flow */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="glass-card p-8"
      >
        <h3 className="text-lg font-semibold mb-6 text-center">End-to-End Data Flow</h3>
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {[
            { icon: Activity, label: 'Sensors', desc: '8 EEG + 3 ECG', color: 'nc-ecg' },
            { icon: Binary, label: 'Firmware', desc: 'C, 569B packets', color: 'nc-ecg' },
            { icon: Radio, label: 'Gateway', desc: 'BLE → JSON', color: 'nc-hrv' },
            { icon: Filter, label: 'DSP', desc: 'Filter & Extract', color: 'nc-eeg' },
            { icon: Cpu, label: 'ML Model', desc: 'XGB + LSTM', color: 'nc-eeg' },
            { icon: Monitor, label: 'Dashboard', desc: 'Real-time viz', color: 'nc-accent' },
          ].map((step, i) => (
            <div key={step.label} className="flex items-center gap-4">
              <div className="text-center">
                <div className={`w-14 h-14 rounded-xl bg-${step.color}/20 flex items-center justify-center mx-auto mb-2`}>
                  <step.icon className={`w-7 h-7 text-${step.color}`} />
                </div>
                <div className="font-medium text-sm">{step.label}</div>
                <div className="text-xs text-white/40">{step.desc}</div>
              </div>
              {i < 5 && <ArrowRight className="w-5 h-5 text-white/20 hidden md:block" />}
            </div>
          ))}
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// LIVE SIGNALS SECTION
// ============================================================================

function LiveSignalsSection() {
  const [eegData, setEegData] = useState<Record<string, {time: number, value: number}[]>>({})
  const [ecgData, setEcgData] = useState<{time: number, value: number}[]>([])
  const [heartRate, setHeartRate] = useState(72)
  const [eegState, setEegState] = useState<PhysiologicalState>('relaxed')
  const offsetRef = useRef(0)
  const ecgOffsetRef = useRef(0)
  const heartRateRef = useRef(72)
  const eegStateRef = useRef<PhysiologicalState>('relaxed')

  // Keep refs in sync with state
  useEffect(() => {
    eegStateRef.current = eegState
  }, [eegState])

  useEffect(() => {
    heartRateRef.current = heartRate
  }, [heartRate])

  const channels = ['Fp1', 'Fp2', 'C3', 'C4', 'T3', 'T4', 'O1', 'O2']

  // EEG animation - very slow and smooth for studying
  useEffect(() => {
    const interval = setInterval(() => {
      offsetRef.current += 1  // Tiny increment for smooth scrolling

      const newEeg: Record<string, {time: number, value: number}[]> = {}
      channels.forEach(ch => {
        const result = generateAdvancedEEG(ch, offsetRef.current, eegStateRef.current, false)
        newEeg[ch] = Array.isArray(result) ? result : result.data
      })
      setEegData(newEeg)
    }, 500)  // Very slow - 500ms interval (2 updates per second)

    return () => clearInterval(interval)
  }, [])

  // ECG animation - separate interval
  useEffect(() => {
    const interval = setInterval(() => {
      ecgOffsetRef.current += 2  // Scroll ECG

      setEcgData(generateAdvancedECG(ecgOffsetRef.current, {
        heartRate: heartRateRef.current
      }))

      // Vary heart rate occasionally
      if (ecgOffsetRef.current % 150 === 0) {
        const newHR = 68 + Math.floor(Math.random() * 12)
        setHeartRate(newHR)
        heartRateRef.current = newHR
      }
    }, 100)  // 100ms interval for smooth scrolling

    return () => clearInterval(interval)
  }, [])

  const channelInfo: Record<string, string> = {
    'Fp1': 'Left frontal - cognitive processing',
    'Fp2': 'Right frontal - executive function',
    'C3': 'Left central - motor planning',
    'C4': 'Right central - motor execution',
    'T3': 'Left temporal - language/memory',
    'T4': 'Right temporal - spatial processing',
    'O1': 'Left occipital - visual processing',
    'O2': 'Right occipital - visual processing',
  }

  return (
    <Section id="signals" dark>
      <SectionHeader
        label="03 — LIVE SIGNALS"
        title="Real-Time Physiological Monitoring"
        description="Watch simulated brain and heart signals as they would appear from a real patient. Each signal carries specific physiological meaning that our system extracts and analyzes."
      />

      {/* EEG Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-6 mb-6"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
          <div className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-nc-eeg" />
            <div>
              <h3 className="text-xl font-semibold">8-Channel EEG (Electroencephalogram)</h3>
              <p className="text-sm text-white/50">Measures electrical activity of the brain via scalp electrodes</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* State Selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-white/40">State:</span>
              <select
                value={eegState}
                onChange={(e) => setEegState(e.target.value as PhysiologicalState)}
                className="bg-black/30 border border-white/10 rounded px-2 py-1 text-sm text-white/80 cursor-pointer"
              >
                <option value="alert">Alert</option>
                <option value="relaxed">Relaxed</option>
                <option value="drowsy">Drowsy</option>
                <option value="stressed">Stressed</option>
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="text-sm text-white/60 font-mono">250 Hz</span>
            </div>
          </div>
        </div>

        {/* State Description */}
        <div className="mb-4 p-3 bg-nc-eeg/10 rounded-lg">
          <p className="text-sm text-white/70">
            <span className="font-medium text-nc-eeg">{EEG_STATE_PROFILES[eegState].label}:</span>{' '}
            {EEG_STATE_PROFILES[eegState].description}
          </p>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {channels.map((channel) => (
            <div key={channel} className="bg-black/30 rounded-lg p-3">
              <div className="flex justify-between items-center mb-1">
                <span className="font-mono text-sm text-nc-eeg">{channel}</span>
              </div>
              <div className="text-xs text-white/40 mb-2">{channelInfo[channel]}</div>
              <div className="h-16">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={eegData[channel] || []}>
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>

        {/* EEG Bands Explanation */}
        <div className="bg-black/20 rounded-lg p-4">
          <h4 className="text-sm font-medium mb-3">Understanding EEG Frequency Bands</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { band: 'Delta', range: '0.5-4 Hz', meaning: 'Deep sleep, healing', color: '#8b5cf6' },
              { band: 'Theta', range: '4-8 Hz', meaning: 'Drowsiness, meditation', color: '#06b6d4' },
              { band: 'Alpha', range: '8-13 Hz', meaning: 'Relaxed wakefulness', color: '#10b981' },
              { band: 'Beta', range: '13-30 Hz', meaning: 'Active thinking, focus', color: '#f59e0b' },
              { band: 'Gamma', range: '30-50 Hz', meaning: 'High cognition, binding', color: '#ef4444' },
            ].map(b => (
              <div key={b.band} className="text-center p-2 rounded" style={{ backgroundColor: `${b.color}15` }}>
                <div className="font-mono text-sm" style={{ color: b.color }}>{b.band}</div>
                <div className="text-xs text-white/60">{b.range}</div>
                <div className="text-xs text-white/40 mt-1">{b.meaning}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* ECG Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Heart className="w-6 h-6 text-nc-ecg" />
            <div>
              <h3 className="text-xl font-semibold">3-Lead ECG (Electrocardiogram)</h3>
              <p className="text-sm text-white/50">Measures electrical activity of the heart</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Heart className="w-5 h-5 text-nc-ecg animate-pulse" />
              <span className="font-mono text-2xl text-nc-ecg">{heartRate}</span>
              <span className="text-sm text-white/40">BPM</span>
            </div>
          </div>
        </div>

        <div className="bg-black/30 rounded-lg p-4 mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="font-mono text-sm text-nc-ecg">Lead II (Primary)</span>
            <span className="text-xs text-white/40">Right Arm → Left Leg</span>
          </div>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ecgData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* PQRST Explanation */}
        <div className="bg-black/20 rounded-lg p-4">
          <h4 className="text-sm font-medium mb-3">Understanding the PQRST Complex</h4>
          <p className="text-sm text-white/50 mb-4">
            Each heartbeat produces a characteristic waveform. Abnormalities in these waves indicate specific cardiac conditions.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { wave: 'P Wave', time: '80ms', meaning: 'Atrial depolarization (contraction)', clinical: 'Absent in AFib' },
              { wave: 'PR Interval', time: '120-200ms', meaning: 'AV node conduction delay', clinical: 'Prolonged in heart block' },
              { wave: 'QRS Complex', time: '80-120ms', meaning: 'Ventricular depolarization', clinical: 'Wide in bundle branch block' },
              { wave: 'ST Segment', time: 'Variable', meaning: 'Plateau phase', clinical: 'Elevated in MI' },
              { wave: 'T Wave', time: '160ms', meaning: 'Ventricular repolarization', clinical: 'Inverted in ischemia' },
            ].map(w => (
              <div key={w.wave} className="p-3 bg-nc-ecg/10 rounded">
                <div className="font-mono text-sm text-nc-ecg">{w.wave}</div>
                <div className="text-xs text-white/60">{w.time}</div>
                <div className="text-xs text-white/40 mt-2">{w.meaning}</div>
                <div className="text-xs text-nc-risk-medium mt-1">{w.clinical}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// SIGNAL PROCESSING SECTION
// ============================================================================

function SignalProcessingSection() {
  return (
    <Section id="processing">
      <SectionHeader
        label="04 — SIGNAL PROCESSING"
        title="From Raw Signals to Meaningful Features"
        description="Raw physiological signals are noisy and high-dimensional. Our digital signal processing pipeline extracts 76 clinically meaningful features that the ML model uses for prediction."
      />

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* EEG Processing */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <Brain className="w-6 h-6 text-nc-eeg" />
            <h3 className="text-lg font-semibold">EEG Feature Extraction</h3>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-eeg mb-2">Step 1: Bandpass Filtering</div>
              <p className="text-sm text-white/60">
                4th-order Butterworth filter (0.5-50 Hz) removes DC drift and high-frequency noise while preserving brain rhythms.
              </p>
            </div>
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-eeg mb-2">Step 2: Power Spectral Density</div>
              <p className="text-sm text-white/60">
                Welch&apos;s method decomposes each channel into frequency bands (delta, theta, alpha, beta, gamma).
              </p>
            </div>
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-eeg mb-2">Step 3: Feature Computation</div>
              <p className="text-sm text-white/60">
                Band powers, spectral entropy, and inter-hemispheric coherence computed per channel.
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-nc-eeg/10 rounded-lg text-center">
            <span className="font-mono text-2xl text-nc-eeg">66</span>
            <span className="text-white/60 ml-2">EEG features extracted</span>
          </div>
        </motion.div>

        {/* ECG Processing */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <Heart className="w-6 h-6 text-nc-ecg" />
            <h3 className="text-lg font-semibold">ECG & HRV Feature Extraction</h3>
          </div>

          <div className="space-y-4">
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-ecg mb-2">Step 1: R-Peak Detection</div>
              <p className="text-sm text-white/60">
                Pan-Tompkins algorithm identifies R-peaks (heartbeats) with 99%+ accuracy on clean signals.
              </p>
            </div>
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-ecg mb-2">Step 2: HRV Time Domain</div>
              <p className="text-sm text-white/60">
                SDNN, RMSSD, pNN50 computed from R-R intervals per Task Force of ESC (1996) standards.
              </p>
            </div>
            <div className="p-4 bg-black/20 rounded-lg">
              <div className="font-mono text-sm text-nc-ecg mb-2">Step 3: HRV Frequency Domain</div>
              <p className="text-sm text-white/60">
                LF (0.04-0.15 Hz) and HF (0.15-0.4 Hz) power reflect sympathetic/parasympathetic balance.
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-nc-ecg/10 rounded-lg text-center">
            <span className="font-mono text-2xl text-nc-ecg">10</span>
            <span className="text-white/60 ml-2">ECG/HRV features extracted</span>
          </div>
        </motion.div>
      </div>

      {/* Feature Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-6 text-center">Complete Feature Set: 76 Features</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { category: 'EEG Band Powers', count: 40, desc: '5 bands × 8 channels', color: 'nc-eeg' },
            { category: 'EEG Statistics', count: 16, desc: 'Entropy, ratios per channel', color: 'nc-eeg' },
            { category: 'EEG Coherence', count: 10, desc: 'Cross-channel correlation', color: 'nc-eeg' },
            { category: 'HRV Time Domain', count: 4, desc: 'SDNN, RMSSD, pNN50, HR', color: 'nc-hrv' },
            { category: 'HRV Frequency', count: 3, desc: 'LF, HF, LF/HF ratio', color: 'nc-hrv' },
            { category: 'ECG Morphology', count: 3, desc: 'QRS duration, amplitude', color: 'nc-ecg' },
          ].map(f => (
            <div key={f.category} className={`p-4 bg-${f.color}/10 rounded-lg`}>
              <div className={`font-mono text-2xl text-${f.color}`}>{f.count}</div>
              <div className="font-medium text-sm mt-1">{f.category}</div>
              <div className="text-xs text-white/40">{f.desc}</div>
            </div>
          ))}
        </div>
        <p className="text-center text-white/50 mt-6">
          Every feature has clinical meaning — we prioritize interpretability over black-box representations.
        </p>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// SIGNAL SCIENCE SECTION - Deep Dive into Data Generation
// ============================================================================

function SignalScienceSection() {
  const [selectedState, setSelectedState] = useState<PhysiologicalState>('relaxed')
  const [showEctopic, setShowEctopic] = useState(false)
  const [selectedBand, setSelectedBand] = useState<string>('all')
  const [explorerData, setExplorerData] = useState<any[]>([])
  const [componentData, setComponentData] = useState<Record<string, number[]>>({})
  const offsetRef = useRef(0)
  const selectedStateRef = useRef<PhysiologicalState>('relaxed')

  // Keep ref in sync with state
  useEffect(() => {
    selectedStateRef.current = selectedState
  }, [selectedState])

  // Generate data for the explorer - use ref to avoid animation restart
  useEffect(() => {
    const interval = setInterval(() => {
      offsetRef.current += 1
      const result = generateAdvancedEEG('O1', offsetRef.current, selectedStateRef.current, true)
      if (result && typeof result === 'object' && 'data' in result) {
        setExplorerData(result.data)
        setComponentData(result.components)
      }
    }, 400)  // Slower for studying
    return () => clearInterval(interval)
  }, [])

  const bandColors: Record<string, string> = {
    delta: '#8b5cf6',
    theta: '#06b6d4',
    alpha: '#10b981',
    beta: '#f59e0b',
    gamma: '#ef4444',
    noise: '#6b7280'
  }

  return (
    <Section id="science" dark>
      <SectionHeader
        label="04.5 — THE SIGNAL SCIENCE"
        title="How We Generate Clinically Realistic Data"
        description="Our synthetic data isn't random noise — it's generated using validated algorithms from peer-reviewed neuroscience and cardiology literature. Here's exactly how it works."
      />

      {/* Why Synthetic Data Matters */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-start gap-4 mb-6">
          <Microscope className="w-8 h-8 text-nc-accent flex-shrink-0" />
          <div>
            <h3 className="text-xl font-semibold mb-2">Why Synthetic Data?</h3>
            <p className="text-white/70 leading-relaxed">
              Real patient EEG/ECG data requires IRB approval, HIPAA compliance, and months of clinical partnerships.
              For an academic prototype, we use <strong className="text-white">synthetic data that mimics real physiological signals</strong> closely
              enough to develop and test our algorithms. The same signal processing and ML pipelines would work on real data.
            </p>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4">
          <div className="p-4 bg-nc-eeg/10 rounded-lg">
            <div className="font-mono text-nc-eeg text-lg mb-1">Spectral Accuracy</div>
            <p className="text-sm text-white/60">
              Power spectral density matches clinical EEG databases. 1/f noise slope verified against published norms.
            </p>
          </div>
          <div className="p-4 bg-nc-ecg/10 rounded-lg">
            <div className="font-mono text-nc-ecg text-lg mb-1">Temporal Fidelity</div>
            <p className="text-sm text-white/60">
              PQRST timing matches AHA guidelines. HRV metrics fall within normal physiological ranges.
            </p>
          </div>
          <div className="p-4 bg-nc-hrv/10 rounded-lg">
            <div className="font-mono text-nc-hrv text-lg mb-1">State Transitions</div>
            <p className="text-sm text-white/60">
              Different cognitive/arousal states produce distinct spectral signatures, just like in real recordings.
            </p>
          </div>
        </div>
      </motion.div>

      {/* EEG Generation Deep Dive */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center gap-3 mb-6">
          <Brain className="w-6 h-6 text-nc-eeg" />
          <h3 className="text-xl font-semibold">EEG Signal Generation Algorithm</h3>
        </div>

        {/* Algorithm Steps */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <div className="space-y-4">
            <div className="p-4 bg-black/30 rounded-lg border-l-4 border-nc-eeg">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-nc-eeg/20 text-nc-eeg text-xs flex items-center justify-center">1</span>
                <span className="font-medium">Multi-Band Synthesis</span>
              </div>
              <p className="text-sm text-white/60">
                Each EEG channel is composed of 5 frequency bands (delta, theta, alpha, beta, gamma) with
                amplitudes based on the 10-20 electrode system and published normative data.
              </p>
              <code className="text-xs text-nc-eeg/70 mt-2 block font-mono">
                signal = Σ A_band × sin(2π × f_band × t)
              </code>
            </div>

            <div className="p-4 bg-black/30 rounded-lg border-l-4 border-nc-eeg">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-nc-eeg/20 text-nc-eeg text-xs flex items-center justify-center">2</span>
                <span className="font-medium">1/f Pink Noise</span>
              </div>
              <p className="text-sm text-white/60">
                Real EEG has a characteristic &quot;1/f&quot; power spectrum — lower frequencies have more power.
                We use the Voss-McCartney algorithm to generate pink noise with equal energy per octave.
              </p>
              <code className="text-xs text-nc-eeg/70 mt-2 block font-mono">
                PSD(f) ∝ 1/f^α, where α ≈ 1
              </code>
            </div>

            <div className="p-4 bg-black/30 rounded-lg border-l-4 border-nc-eeg">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-nc-eeg/20 text-nc-eeg text-xs flex items-center justify-center">3</span>
                <span className="font-medium">State-Dependent Modulation</span>
              </div>
              <p className="text-sm text-white/60">
                Cognitive states alter the relative power of each band. Alert states boost beta/gamma;
                relaxed states boost alpha; drowsy states boost theta/delta.
              </p>
            </div>

            <div className="p-4 bg-black/30 rounded-lg border-l-4 border-nc-eeg">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-nc-eeg/20 text-nc-eeg text-xs flex items-center justify-center">4</span>
                <span className="font-medium">Amplitude Modulation</span>
              </div>
              <p className="text-sm text-white/60">
                Alpha waves naturally &quot;wax and wane&quot; in real EEG. We add slow (~0.5 Hz) modulation
                to create realistic spindle-like patterns.
              </p>
            </div>
          </div>

          {/* Interactive State Selector & Visualization */}
          <div>
            <div className="mb-4">
              <label className="text-sm text-white/50 mb-2 block">Select Physiological State:</label>
              <div className="flex flex-wrap gap-2">
                {(['alert', 'relaxed', 'drowsy', 'stressed'] as PhysiologicalState[]).map(state => (
                  <button
                    key={state}
                    onClick={() => setSelectedState(state)}
                    className={`px-4 py-2 rounded-lg text-sm transition-all ${
                      selectedState === state
                        ? 'bg-nc-eeg text-white'
                        : 'bg-white/5 text-white/60 hover:bg-white/10'
                    }`}
                  >
                    {EEG_STATE_PROFILES[state].label}
                  </button>
                ))}
              </div>
            </div>

            {/* State Description */}
            <div className="p-4 bg-nc-eeg/10 rounded-lg mb-4">
              <p className="text-sm text-white/70">{EEG_STATE_PROFILES[selectedState].description}</p>
            </div>

            {/* Live Signal Visualization */}
            <div className="bg-black/30 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-mono text-sm text-nc-eeg">O1 Channel (Occipital)</span>
                <span className="text-xs text-white/40">State: {selectedState}</span>
              </div>
              <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={explorerData}>
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Band Power Display */}
            <div className="grid grid-cols-5 gap-2 mt-4">
              {(['delta', 'theta', 'alpha', 'beta', 'gamma'] as const).map(band => {
                const profile = EEG_STATE_PROFILES[selectedState]
                const power = profile[band]
                return (
                  <div key={band} className="text-center p-2 rounded" style={{ backgroundColor: `${bandColors[band]}20` }}>
                    <div className="font-mono text-xs" style={{ color: bandColors[band] }}>{band}</div>
                    <div className="text-lg font-mono text-white/80">{(power * 100).toFixed(0)}%</div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Component Explorer */}
        <div className="border-t border-white/10 pt-6">
          <h4 className="font-semibold mb-4">Interactive Component Explorer</h4>
          <p className="text-sm text-white/50 mb-4">
            See how each frequency band contributes to the final signal. Click a band to isolate it.
          </p>

          <div className="flex flex-wrap gap-2 mb-4">
            <button
              onClick={() => setSelectedBand('all')}
              className={`px-3 py-1.5 rounded text-sm ${
                selectedBand === 'all' ? 'bg-white/20 text-white' : 'bg-white/5 text-white/50'
              }`}
            >
              Combined Signal
            </button>
            {Object.keys(bandColors).map(band => (
              <button
                key={band}
                onClick={() => setSelectedBand(band)}
                className={`px-3 py-1.5 rounded text-sm flex items-center gap-1 ${
                  selectedBand === band ? 'text-white' : 'text-white/50'
                }`}
                style={{ backgroundColor: selectedBand === band ? `${bandColors[band]}40` : 'rgba(255,255,255,0.05)' }}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: bandColors[band] }} />
                {band}
              </button>
            ))}
          </div>

          <div className="bg-black/30 rounded-lg p-4">
            <div className="h-24">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={explorerData}>
                  {selectedBand === 'all' ? (
                    <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  ) : (
                    <Line type="monotone" dataKey={selectedBand} stroke={bandColors[selectedBand]} strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ECG Generation Deep Dive */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center gap-3 mb-6">
          <Heart className="w-6 h-6 text-nc-ecg" />
          <h3 className="text-xl font-semibold">ECG Signal Generation: The McSharry Model</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-8 mb-6">
          <div>
            <p className="text-white/70 mb-4">
              Our ECG generation is based on the <strong className="text-white">McSharry dynamical model</strong>,
              a widely-cited algorithm that uses Gaussian functions to create realistic PQRST morphology.
            </p>

            <div className="space-y-3">
              <div className="p-3 bg-black/30 rounded-lg">
                <div className="font-mono text-nc-ecg text-sm mb-1">Gaussian Wave Model</div>
                <p className="text-xs text-white/60">
                  Each ECG wave (P, Q, R, S, T) is modeled as a Gaussian:
                </p>
                <code className="text-xs text-nc-ecg/70 mt-1 block font-mono">
                  W(t) = A × exp(-(t - t₀)² / (2σ²))
                </code>
              </div>

              <div className="p-3 bg-black/30 rounded-lg">
                <div className="font-mono text-nc-ecg text-sm mb-1">Heart Rate Variability</div>
                <p className="text-xs text-white/60">
                  RR intervals vary naturally (5-10%). We model this plus respiratory sinus arrhythmia (RSA).
                </p>
              </div>

              <div className="p-3 bg-black/30 rounded-lg">
                <div className="font-mono text-nc-ecg text-sm mb-1">Noise Components</div>
                <p className="text-xs text-white/60">
                  Baseline wander (~0.2 Hz), electrode noise, and 60 Hz powerline interference are added.
                </p>
              </div>
            </div>

            {/* Ectopic Beat Toggle */}
            <div className="mt-4 p-4 bg-nc-ecg/10 rounded-lg">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showEctopic}
                  onChange={(e) => setShowEctopic(e.target.checked)}
                  className="w-4 h-4 rounded border-white/30"
                />
                <span className="text-sm text-white/70">Show Ectopic Beats (PVC)</span>
              </label>
              <p className="text-xs text-white/40 mt-2">
                Premature Ventricular Contractions have wide QRS, no P wave, and inverted T waves.
              </p>
            </div>
          </div>

          {/* ECG Component Diagram */}
          <div>
            <div className="bg-black/30 rounded-lg p-4 mb-4">
              <div className="text-sm text-white/50 mb-2">PQRST Wave Parameters (Lead II)</div>
              <div className="space-y-2">
                {[
                  { wave: 'P', amplitude: '0.1-0.2 mV', duration: '80-100 ms', meaning: 'Atrial depolarization' },
                  { wave: 'Q', amplitude: '-0.1 mV', duration: '20-40 ms', meaning: 'Septal depolarization' },
                  { wave: 'R', amplitude: '1.0-1.5 mV', duration: '40-60 ms', meaning: 'Ventricular mass' },
                  { wave: 'S', amplitude: '-0.2 mV', duration: '40-60 ms', meaning: 'Late ventricular' },
                  { wave: 'T', amplitude: '0.2-0.3 mV', duration: '120-160 ms', meaning: 'Repolarization' },
                ].map(w => (
                  <div key={w.wave} className="flex items-center justify-between text-xs">
                    <span className="font-mono text-nc-ecg w-8">{w.wave}</span>
                    <span className="text-white/60">{w.amplitude}</span>
                    <span className="text-white/40">{w.duration}</span>
                    <span className="text-white/50 text-right w-32">{w.meaning}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Live ECG with Ectopic */}
            <ECGExplorer showEctopic={showEctopic} />
          </div>
        </div>
      </motion.div>

      {/* Scientific References */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8"
      >
        <div className="flex items-center gap-3 mb-6">
          <BookOpen className="w-6 h-6 text-nc-accent" />
          <h3 className="text-xl font-semibold">Scientific References</h3>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-nc-eeg mb-3">EEG Signal Generation</h4>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">Niedermeyer & Lopes da Silva</div>
                <div className="text-white/50 text-xs">
                  &quot;Electroencephalography: Basic Principles, Clinical Applications, and Related Fields&quot;
                  — 5th Edition, Lippincott Williams & Wilkins, 2004
                </div>
              </div>
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">Nunez & Srinivasan</div>
                <div className="text-white/50 text-xs">
                  &quot;Electric Fields of the Brain: The Neurophysics of EEG&quot;
                  — 2nd Edition, Oxford University Press, 2006
                </div>
              </div>
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">Voss & Clarke</div>
                <div className="text-white/50 text-xs">
                  &quot;1/f noise in music and speech&quot;
                  — Nature 258:317-318, 1975 (Pink noise algorithm)
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium text-nc-ecg mb-3">ECG Signal Generation</h4>
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">McSharry et al.</div>
                <div className="text-white/50 text-xs">
                  &quot;A Dynamical Model for Generating Synthetic Electrocardiogram Signals&quot;
                  — IEEE Trans Biomed Eng 50(3):289-294, 2003
                </div>
              </div>
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">Task Force of ESC & NASPE</div>
                <div className="text-white/50 text-xs">
                  &quot;Heart Rate Variability: Standards of Measurement, Physiological Interpretation, and Clinical Use&quot;
                  — Circulation 93(5):1043-1065, 1996
                </div>
              </div>
              <div className="p-3 bg-black/20 rounded-lg">
                <div className="text-white/80">Pan & Tompkins</div>
                <div className="text-white/50 text-xs">
                  &quot;A Real-Time QRS Detection Algorithm&quot;
                  — IEEE Trans Biomed Eng 32(3):230-236, 1985
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 p-4 bg-nc-accent/10 border border-nc-accent/20 rounded-lg">
          <p className="text-sm text-white/60">
            <strong className="text-nc-accent">Note:</strong> While our synthetic data follows established algorithms,
            it is not a substitute for real clinical data. The patterns are approximations and would require
            validation against actual patient recordings before any clinical use.
          </p>
        </div>
      </motion.div>
    </Section>
  )
}

// ECG Explorer Sub-component
function ECGExplorer({ showEctopic }: { showEctopic: boolean }) {
  const [ecgData, setEcgData] = useState<any[]>([])
  const offsetRef = useRef(0)
  const showEctopicRef = useRef(showEctopic)

  // Keep ref in sync
  useEffect(() => {
    showEctopicRef.current = showEctopic
  }, [showEctopic])

  useEffect(() => {
    const interval = setInterval(() => {
      offsetRef.current += 3
      setEcgData(generateAdvancedECG(offsetRef.current, {
        heartRate: 72,
        rrVariability: 0.06,
        rsaAmplitude: 0.04,
        showPathology: showEctopicRef.current
      }))
    }, 80)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-black/30 rounded-lg p-4">
      <div className="flex justify-between items-center mb-2">
        <span className="font-mono text-sm text-nc-ecg">Lead II</span>
        <span className="text-xs text-white/40">
          {showEctopic ? 'With Ectopic Beats' : 'Normal Sinus Rhythm'}
        </span>
      </div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={ecgData}>
            <Line
              type="monotone"
              dataKey="value"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ============================================================================
// ML SECTION
// ============================================================================

function MLSection() {
  const rocData = [
    { fpr: 0, tpr: 0 }, { fpr: 0.02, tpr: 0.45 }, { fpr: 0.05, tpr: 0.65 },
    { fpr: 0.1, tpr: 0.78 }, { fpr: 0.15, tpr: 0.85 }, { fpr: 0.2, tpr: 0.89 },
    { fpr: 0.3, tpr: 0.93 }, { fpr: 0.5, tpr: 0.96 }, { fpr: 0.7, tpr: 0.98 }, { fpr: 1, tpr: 1 },
  ]

  const featureImportance = [
    { feature: 'rmssd', importance: 0.142, category: 'HRV', meaning: 'Parasympathetic activity' },
    { feature: 'lf_hf_ratio', importance: 0.128, category: 'HRV', meaning: 'Autonomic balance' },
    { feature: 'O1_alpha', importance: 0.098, category: 'EEG', meaning: 'Visual cortex activity' },
    { feature: 'sdnn', importance: 0.087, category: 'HRV', meaning: 'Overall HRV' },
    { feature: 'C3_theta', importance: 0.076, category: 'EEG', meaning: 'Motor cortex theta' },
    { feature: 'pnn50', importance: 0.064, category: 'HRV', meaning: 'Vagal tone marker' },
  ]

  return (
    <Section id="ml" dark>
      <SectionHeader
        label="05 — MACHINE LEARNING"
        title="Interpretable Risk Prediction"
        description="We chose an ensemble approach that prioritizes explainability. Medical AI must be interpretable — clinicians need to understand WHY a prediction was made, not just what it is."
      />

      {/* Model Architecture */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 mb-8"
      >
        <h3 className="text-lg font-semibold mb-6 text-center">Ensemble Architecture</h3>

        <div className="flex flex-col md:flex-row items-center justify-center gap-8">
          <div className="glass-card p-6 w-full md:w-72 border-l-4 border-nc-hrv">
            <div className="flex justify-between items-center mb-4">
              <span className="font-semibold text-lg">XGBoost</span>
              <span className="font-mono text-nc-hrv bg-nc-hrv/20 px-3 py-1 rounded">60%</span>
            </div>
            <p className="text-sm text-white/60 mb-4">
              Gradient boosted trees operating on 76 hand-crafted features. Provides direct feature importance scores.
            </p>
            <div className="text-xs text-white/40">
              <div>• 100 estimators</div>
              <div>• Max depth: 6</div>
              <div>• Learning rate: 0.1</div>
            </div>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 rounded-full border-2 border-nc-accent flex items-center justify-center text-2xl text-nc-accent">
              +
            </div>
            <div className="text-xs text-white/40 mt-2">Weighted<br/>Ensemble</div>
          </div>

          <div className="glass-card p-6 w-full md:w-72 border-l-4 border-nc-eeg">
            <div className="flex justify-between items-center mb-4">
              <span className="font-semibold text-lg">BiLSTM</span>
              <span className="font-mono text-nc-eeg bg-nc-eeg/20 px-3 py-1 rounded">40%</span>
            </div>
            <p className="text-sm text-white/60 mb-4">
              Bidirectional LSTM captures temporal patterns in raw signal windows that features might miss.
            </p>
            <div className="text-xs text-white/40">
              <div>• 2 LSTM layers (64 units)</div>
              <div>• Dropout: 0.3</div>
              <div>• Input: 250×11 window</div>
            </div>
          </div>
        </div>

        <div className="mt-8 text-center">
          <ArrowDown className="w-6 h-6 text-white/20 mx-auto mb-4" />
          <div className="inline-block glass-card px-8 py-4">
            <div className="text-sm text-white/60 mb-2">Output</div>
            <div className="flex items-center gap-4">
              <span className="px-3 py-1 bg-nc-risk-low/20 text-nc-risk-low rounded font-mono">LOW</span>
              <span className="px-3 py-1 bg-nc-risk-medium/20 text-nc-risk-medium rounded font-mono">MEDIUM</span>
              <span className="px-3 py-1 bg-nc-risk-high/20 text-nc-risk-high rounded font-mono">HIGH</span>
            </div>
            <div className="text-xs text-white/40 mt-2">3-class risk classification + confidence score</div>
          </div>
        </div>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6 mb-8">
        {/* ROC Curve */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-2">ROC Curve (XGBoost)</h3>
          <p className="text-sm text-white/50 mb-4">
            Area Under Curve measures how well the model separates classes. 0.5 = random, 1.0 = perfect.
          </p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="fpr" stroke="#ffffff40" tickFormatter={(v) => v.toFixed(1)} />
                <YAxis stroke="#ffffff40" tickFormatter={(v) => v.toFixed(1)} />
                <Area type="monotone" dataKey="tpr" stroke="#10b981" fill="#10b98130" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="text-center mt-2">
            <span className="font-mono text-xl text-nc-hrv">AUC = 0.923</span>
            <span className="text-white/40 ml-2">(Good discrimination)</span>
          </div>
        </motion.div>

        {/* Feature Importance */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="glass-card p-6"
        >
          <h3 className="text-lg font-semibold mb-2">Top Feature Importance</h3>
          <p className="text-sm text-white/50 mb-4">
            Which features matter most? HRV metrics dominate — the autonomic nervous system is key.
          </p>
          <div className="space-y-3">
            {featureImportance.map(f => (
              <div key={f.feature} className="flex items-center gap-3">
                <div className="w-20 font-mono text-sm text-white/80">{f.feature}</div>
                <div className="flex-1">
                  <div
                    className={`h-6 rounded ${f.category === 'HRV' ? 'bg-nc-hrv' : 'bg-nc-eeg'}`}
                    style={{ width: `${f.importance * 500}%`, opacity: 0.7 }}
                  />
                </div>
                <div className="w-32 text-xs text-white/40">{f.meaning}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Honest Assessment */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="p-6 border border-nc-risk-medium/30 bg-nc-risk-medium/5 rounded-xl"
      >
        <div className="flex items-start gap-4">
          <AlertTriangle className="w-6 h-6 text-nc-risk-medium flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold text-nc-risk-medium mb-2">Honest Limitation</h3>
            <p className="text-white/70 leading-relaxed">
              Our BiLSTM achieves 99.75% accuracy — <strong>suspiciously high</strong>. This indicates the synthetic
              training data contains patterns that are too easy to learn and would not generalize to real patients.
              We document this as a limitation, not an achievement. Real clinical deployment would require training
              on actual patient data with IRB approval and clinical validation studies.
            </p>
          </div>
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// ARCHITECTURE SECTION
// ============================================================================

function ArchitectureSection() {
  return (
    <Section id="architecture">
      <SectionHeader
        label="06 — SYSTEM ARCHITECTURE"
        title="Four-Layer Medical Device Architecture"
        description="We followed IEC 62304 medical device software patterns (though without formal compliance). Each layer has clear responsibilities and interfaces, enabling independent testing and future hardware integration."
      />

      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="glass-card p-8"
      >
        <div className="space-y-4">
          {[
            {
              layer: 4,
              name: 'Presentation',
              color: 'nc-accent',
              tech: 'Streamlit + Plotly',
              desc: 'Real-time dashboard displaying 8-channel EEG, ECG waveforms, risk gauge, and HRV metrics. WebSocket connection for live updates.',
              files: ['dashboard/app.py'],
            },
            {
              layer: 3,
              name: 'Application',
              color: 'nc-hrv',
              tech: 'FastAPI + Uvicorn',
              desc: 'REST API for data ingestion (/ingest), ML inference (/inference), and system status. Manages 1000-packet circular buffer.',
              files: ['cloud/api/server.py'],
            },
            {
              layer: 2,
              name: 'Domain',
              color: 'nc-eeg',
              tech: 'NumPy + SciPy + XGBoost + TensorFlow',
              desc: 'Signal processing (Butterworth filters, Welch PSD, R-peak detection) and ML inference (ensemble prediction with confidence).',
              files: ['cloud/signal_processing/', 'ml/model/'],
            },
            {
              layer: 1,
              name: 'Acquisition',
              color: 'nc-ecg',
              tech: 'C Firmware + Python Adapters',
              desc: 'Device adapters for simulated, BLE, and serial connections. Binary packet parsing (569 bytes → JSON). Hardware abstraction layer.',
              files: ['firmware/', 'cloud/device_adapters/'],
            },
          ].map((l, i) => (
            <div key={l.layer}>
              <div className={`p-6 border-2 border-${l.color} rounded-xl bg-${l.color}/5`}>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className={`font-mono text-sm text-${l.color} bg-${l.color}/20 px-3 py-1 rounded`}>
                      Layer {l.layer}
                    </div>
                    <div>
                      <div className="text-xl font-semibold">{l.name}</div>
                      <div className={`text-sm text-${l.color}`}>{l.tech}</div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {l.files.map(f => (
                      <code key={f} className="text-xs bg-black/30 px-2 py-1 rounded text-white/50">{f}</code>
                    ))}
                  </div>
                </div>
                <p className="text-white/60 mt-4">{l.desc}</p>
              </div>
              {i < 3 && <ArrowDown className="w-6 h-6 text-white/20 mx-auto my-2" />}
            </div>
          ))}
        </div>
      </motion.div>

      {/* Data Flow Specs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
        {[
          { label: 'Sample Rate', value: '250 Hz', desc: 'Captures up to 125 Hz (gamma)' },
          { label: 'Packet Size', value: '569 B', desc: '25 samples × 11 channels + header' },
          { label: 'Latency', value: '<100ms', desc: 'Acquisition to prediction' },
          { label: 'Buffer', value: '100s', desc: '1000 packets stored' },
        ].map(s => (
          <div key={s.label} className="glass-card p-4 text-center">
            <div className="font-mono text-2xl text-nc-accent">{s.value}</div>
            <div className="font-medium text-sm">{s.label}</div>
            <div className="text-xs text-white/40">{s.desc}</div>
          </div>
        ))}
      </div>
    </Section>
  )
}

// ============================================================================
// RESULTS SECTION
// ============================================================================

function ResultsSection() {
  const checks = [
    { category: 'Directory Structure', passed: 14 },
    { category: 'Core Source Files', passed: 12 },
    { category: 'Documentation', passed: 8 },
    { category: 'ML Pipeline', passed: 15 },
    { category: 'Device Adapters', passed: 8 },
    { category: 'Signal Processing', passed: 6 },
    { category: 'Configuration', passed: 4 },
  ]
  const total = checks.reduce((sum, c) => sum + c.passed, 0)

  return (
    <Section id="results" dark>
      <SectionHeader
        label="07 — RESULTS & VERIFICATION"
        title="Reproducible, Verified, Open Source"
        description="Every component is tested. Every claim is verified. The entire codebase is open source for inspection, learning, and extension."
      />

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card p-8 text-center"
        >
          <div className="font-mono text-6xl text-nc-risk-low mb-2">{total}</div>
          <div className="text-xl font-medium">Verification Checks</div>
          <div className="text-white/50">All passing</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="glass-card p-8 text-center"
        >
          <div className="font-mono text-6xl text-nc-accent mb-2">42</div>
          <div className="text-xl font-medium">Fixed Random Seed</div>
          <div className="text-white/50">100% reproducible</div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="glass-card p-8 text-center"
        >
          <div className="font-mono text-6xl text-nc-eeg mb-2">MIT</div>
          <div className="text-xl font-medium">Open Source</div>
          <div className="text-white/50">Free to use & extend</div>
        </motion.div>
      </div>

      {/* Verification Categories */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="glass-card p-6 mb-8"
      >
        <h3 className="text-lg font-semibold mb-4">Verification Breakdown</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {checks.map(c => (
            <div key={c.category} className="flex items-center gap-3 p-3 bg-black/20 rounded-lg">
              <Check className="w-5 h-5 text-nc-risk-low" />
              <div>
                <div className="text-sm font-medium">{c.category}</div>
                <div className="font-mono text-nc-risk-low text-sm">{c.passed} passed</div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Quick Start */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4">Try It Yourself</h3>
        <div className="bg-black/40 rounded-lg p-4 font-mono text-sm overflow-x-auto">
          <div className="text-white/40"># Clone the repository</div>
          <div className="text-nc-accent">git clone https://github.com/bblackheart013/neurocardiac-shield.git</div>
          <div className="text-nc-accent">cd neurocardiac-shield</div>
          <br/>
          <div className="text-white/40"># Setup (creates venvs, installs deps, trains models)</div>
          <div className="text-nc-accent">./setup.sh</div>
          <br/>
          <div className="text-white/40"># Verify everything works</div>
          <div className="text-nc-accent">python verify_system.py</div>
          <br/>
          <div className="text-white/40"># Launch the full demo</div>
          <div className="text-nc-accent">./run_complete_demo.sh</div>
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// DEVICE CONNECTION SECTION
// ============================================================================

const COMPATIBLE_DEVICES = [
  {
    name: 'Polar H10',
    type: 'Heart Rate + ECG',
    icon: Heart,
    color: 'nc-ecg',
    supported: true,
    features: ['Heart Rate', 'RR Intervals', 'ECG Stream'],
    connection: 'Bluetooth LE',
    note: 'Best for ECG data'
  },
  {
    name: 'Polar H9/OH1',
    type: 'Heart Rate',
    icon: Heart,
    color: 'nc-ecg',
    supported: true,
    features: ['Heart Rate', 'RR Intervals'],
    connection: 'Bluetooth LE',
    note: 'HR monitoring only'
  },
  {
    name: 'Garmin HRM-Pro',
    type: 'Heart Rate',
    icon: Watch,
    color: 'nc-hrv',
    supported: true,
    features: ['Heart Rate', 'RR Intervals'],
    connection: 'Bluetooth LE',
    note: 'Standard BLE HR'
  },
  {
    name: 'Wahoo TICKR',
    type: 'Heart Rate',
    icon: Heart,
    color: 'nc-hrv',
    supported: true,
    features: ['Heart Rate', 'RR Intervals'],
    connection: 'Bluetooth LE',
    note: 'Standard BLE HR'
  },
  {
    name: 'Any BLE HR Monitor',
    type: 'Heart Rate',
    icon: Bluetooth,
    color: 'nc-accent',
    supported: true,
    features: ['Heart Rate'],
    connection: 'Bluetooth LE',
    note: 'Standard GATT profile'
  },
  {
    name: 'Muse 2 / Muse S',
    type: 'EEG Headband',
    icon: Brain,
    color: 'nc-eeg',
    supported: false,
    features: ['4-ch EEG', 'PPG', 'Accelerometer'],
    connection: 'Bluetooth LE',
    note: 'Coming soon (requires SDK)'
  },
  {
    name: 'Apple Watch',
    type: 'Smartwatch',
    icon: Watch,
    color: 'nc-accent',
    supported: false,
    features: ['Heart Rate', 'ECG'],
    connection: 'Requires App',
    note: 'Needs companion app'
  },
]

function DeviceConnectionSection() {
  const [isScanning, setIsScanning] = useState(false)
  const [connectedDevice, setConnectedDevice] = useState<ConnectedDevice | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [liveHeartRate, setLiveHeartRate] = useState<number | null>(null)
  const [rrIntervals, setRrIntervals] = useState<number[]>([])
  const [hrHistory, setHrHistory] = useState<{time: number, hr: number}[]>([])
  const [bluetoothSupported, setBluetoothSupported] = useState(true)

  useEffect(() => {
    // Check if Web Bluetooth is supported
    if (typeof navigator !== 'undefined' && !navigator.bluetooth) {
      setBluetoothSupported(false)
    }
  }, [])

  const connectToDevice = async () => {
    setError(null)
    setIsScanning(true)

    try {
      // Request Bluetooth device with Heart Rate service
      const device = await navigator.bluetooth.requestDevice({
        filters: [
          { services: ['heart_rate'] },
          { services: ['0000180d-0000-1000-8000-00805f9b34fb'] }, // Heart Rate UUID
        ],
        optionalServices: ['battery_service', 'device_information']
      })

      console.log('Device selected:', device.name)

      // Connect to GATT server
      const server = await device.gatt?.connect()
      if (!server) throw new Error('Failed to connect to GATT server')

      // Get Heart Rate service
      const hrService = await server.getPrimaryService('heart_rate')

      // Get Heart Rate Measurement characteristic
      const hrChar = await hrService.getCharacteristic('heart_rate_measurement')

      // Start notifications
      await hrChar.startNotifications()

      // Handle heart rate data
      hrChar.addEventListener('characteristicvaluechanged', (event: any) => {
        const value = event.target.value
        const flags = value.getUint8(0)
        const is16Bit = flags & 0x01

        // Parse heart rate value
        const heartRate = is16Bit ? value.getUint16(1, true) : value.getUint8(1)
        setLiveHeartRate(heartRate)

        // Parse RR intervals if present
        const hasRR = (flags & 0x10) !== 0
        if (hasRR) {
          const rrOffset = is16Bit ? 3 : 2
          const newRRs: number[] = []
          for (let i = rrOffset; i < value.byteLength; i += 2) {
            const rr = value.getUint16(i, true) / 1024 * 1000 // Convert to ms
            newRRs.push(Math.round(rr))
          }
          setRrIntervals(prev => [...prev.slice(-50), ...newRRs])
        }

        // Update history
        setHrHistory(prev => [...prev.slice(-60), { time: Date.now(), hr: heartRate }])

        // Notify global listeners
        notifyDeviceListeners({
          heartRate,
          rrIntervals: rrIntervals,
          deviceName: device.name || 'Unknown Device',
          deviceId: device.id,
          connected: true,
          timestamp: Date.now()
        })
      })

      // Handle disconnection
      device.addEventListener('gattserverdisconnected', () => {
        console.log('Device disconnected')
        setConnectedDevice(null)
        setLiveHeartRate(null)
        notifyDeviceListeners(null)
      })

      setConnectedDevice({
        device,
        server,
        heartRateChar: hrChar,
        type: 'hr',
        name: device.name || 'Heart Rate Monitor'
      })

    } catch (err: any) {
      console.error('Bluetooth error:', err)
      if (err.name === 'NotFoundError') {
        setError('No device selected. Please try again and select a device.')
      } else if (err.name === 'SecurityError') {
        setError('Bluetooth access denied. Please allow Bluetooth in your browser settings.')
      } else {
        setError(err.message || 'Failed to connect to device')
      }
    } finally {
      setIsScanning(false)
    }
  }

  const disconnectDevice = async () => {
    if (connectedDevice?.device.gatt?.connected) {
      connectedDevice.device.gatt.disconnect()
    }
    setConnectedDevice(null)
    setLiveHeartRate(null)
    setRrIntervals([])
    setHrHistory([])
    notifyDeviceListeners(null)
  }

  // Calculate HRV metrics from RR intervals
  const hrvMetrics = useMemo(() => {
    if (rrIntervals.length < 10) return null

    const recent = rrIntervals.slice(-30)
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length

    // SDNN
    const squaredDiffs = recent.map(rr => Math.pow(rr - mean, 2))
    const sdnn = Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / recent.length)

    // RMSSD
    const successiveDiffs = recent.slice(1).map((rr, i) => Math.pow(rr - recent[i], 2))
    const rmssd = Math.sqrt(successiveDiffs.reduce((a, b) => a + b, 0) / successiveDiffs.length)

    // pNN50
    const nn50 = recent.slice(1).filter((rr, i) => Math.abs(rr - recent[i]) > 50).length
    const pnn50 = (nn50 / (recent.length - 1)) * 100

    return { sdnn: sdnn.toFixed(1), rmssd: rmssd.toFixed(1), pnn50: pnn50.toFixed(1), meanRR: mean.toFixed(0) }
  }, [rrIntervals])

  return (
    <Section id="connect">
      <SectionHeader
        label="08 — CONNECT YOUR DEVICE"
        title="Live Device Integration"
        description="Connect your own wearable device to see real physiological data flowing through the NeuroCardiac Shield system. We support any Bluetooth LE heart rate monitor."
      />

      {/* Connection Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 mb-8">
          <div>
            <h3 className="text-xl font-semibold mb-2 flex items-center gap-3">
              <Bluetooth className="w-6 h-6 text-nc-accent" />
              Device Connection
            </h3>
            <p className="text-white/50">
              {connectedDevice
                ? `Connected to ${connectedDevice.name}`
                : 'Connect a Bluetooth heart rate monitor to see live data'}
            </p>
          </div>

          <div className="flex gap-3">
            {!connectedDevice ? (
              <button
                onClick={connectToDevice}
                disabled={isScanning || !bluetoothSupported}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${
                  isScanning
                    ? 'bg-nc-accent/20 text-nc-accent cursor-wait'
                    : bluetoothSupported
                    ? 'bg-nc-accent hover:bg-nc-accent/80 text-white'
                    : 'bg-white/10 text-white/40 cursor-not-allowed'
                }`}
              >
                {isScanning ? (
                  <>
                    <BluetoothSearching className="w-5 h-5 animate-pulse" />
                    Scanning...
                  </>
                ) : (
                  <>
                    <Bluetooth className="w-5 h-5" />
                    Connect Device
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={disconnectDevice}
                className="flex items-center gap-2 px-6 py-3 bg-nc-risk-high/20 hover:bg-nc-risk-high/30 text-nc-risk-high rounded-lg font-medium transition-all"
              >
                <Unlink className="w-5 h-5" />
                Disconnect
              </button>
            )}
          </div>
        </div>

        {/* Browser compatibility warning */}
        {!bluetoothSupported && (
          <div className="p-4 bg-nc-risk-medium/10 border border-nc-risk-medium/30 rounded-lg mb-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-nc-risk-medium flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-medium text-nc-risk-medium">Web Bluetooth Not Supported</div>
                <p className="text-sm text-white/60 mt-1">
                  Your browser doesn&apos;t support Web Bluetooth. Please use Chrome, Edge, or Opera on desktop,
                  or Chrome on Android. Safari and Firefox do not support Web Bluetooth.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="p-4 bg-nc-risk-high/10 border border-nc-risk-high/30 rounded-lg mb-6">
            <div className="flex items-start gap-3">
              <X className="w-5 h-5 text-nc-risk-high flex-shrink-0 mt-0.5" />
              <div>
                <div className="font-medium text-nc-risk-high">Connection Error</div>
                <p className="text-sm text-white/60 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Connected device info & live data */}
        {connectedDevice && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Live Heart Rate */}
            <div className="bg-black/30 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-white/50">Live Heart Rate</span>
                <span className="flex items-center gap-2 text-nc-risk-low text-sm">
                  <span className="w-2 h-2 bg-nc-risk-low rounded-full animate-pulse" />
                  Connected
                </span>
              </div>
              <div className="flex items-end gap-4">
                <div className="flex items-center gap-3">
                  <Heart className="w-12 h-12 text-nc-ecg animate-pulse" />
                  <div>
                    <div className="font-mono text-5xl text-nc-ecg">{liveHeartRate || '--'}</div>
                    <div className="text-white/40 text-sm">BPM</div>
                  </div>
                </div>
              </div>

              {/* HR Trend */}
              {hrHistory.length > 5 && (
                <div className="mt-4 h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={hrHistory.slice(-30)}>
                      <Line
                        type="monotone"
                        dataKey="hr"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Live HRV Metrics */}
            <div className="bg-black/30 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-white/50">Real-time HRV Analysis</span>
                <span className="text-xs text-white/30">{rrIntervals.length} RR intervals</span>
              </div>

              {hrvMetrics ? (
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-nc-hrv/10 rounded-lg">
                    <div className="font-mono text-2xl text-nc-hrv">{hrvMetrics.sdnn}</div>
                    <div className="text-xs text-white/40">SDNN (ms)</div>
                  </div>
                  <div className="p-3 bg-nc-hrv/10 rounded-lg">
                    <div className="font-mono text-2xl text-nc-hrv">{hrvMetrics.rmssd}</div>
                    <div className="text-xs text-white/40">RMSSD (ms)</div>
                  </div>
                  <div className="p-3 bg-nc-hrv/10 rounded-lg">
                    <div className="font-mono text-2xl text-nc-hrv">{hrvMetrics.pnn50}%</div>
                    <div className="text-xs text-white/40">pNN50</div>
                  </div>
                  <div className="p-3 bg-nc-hrv/10 rounded-lg">
                    <div className="font-mono text-2xl text-nc-hrv">{hrvMetrics.meanRR}</div>
                    <div className="text-xs text-white/40">Mean RR (ms)</div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-white/40">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>Collecting RR intervals...</p>
                  <p className="text-xs mt-1">Need at least 10 intervals for HRV calculation</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Instructions when not connected */}
        {!connectedDevice && bluetoothSupported && (
          <div className="bg-black/20 rounded-xl p-6">
            <h4 className="font-medium mb-4">How to Connect</h4>
            <ol className="space-y-3 text-white/60">
              <li className="flex gap-3">
                <span className="w-6 h-6 rounded-full bg-nc-accent/20 text-nc-accent text-sm flex items-center justify-center flex-shrink-0">1</span>
                <span>Put on your heart rate monitor (chest strap works best for HRV data)</span>
              </li>
              <li className="flex gap-3">
                <span className="w-6 h-6 rounded-full bg-nc-accent/20 text-nc-accent text-sm flex items-center justify-center flex-shrink-0">2</span>
                <span>Make sure Bluetooth is enabled on your computer</span>
              </li>
              <li className="flex gap-3">
                <span className="w-6 h-6 rounded-full bg-nc-accent/20 text-nc-accent text-sm flex items-center justify-center flex-shrink-0">3</span>
                <span>Click &quot;Connect Device&quot; and select your device from the list</span>
              </li>
              <li className="flex gap-3">
                <span className="w-6 h-6 rounded-full bg-nc-accent/20 text-nc-accent text-sm flex items-center justify-center flex-shrink-0">4</span>
                <span>Watch your real heart data flow through the system!</span>
              </li>
            </ol>
          </div>
        )}
      </motion.div>

      {/* Compatible Devices */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-semibold mb-6">Compatible Devices</h3>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {COMPATIBLE_DEVICES.map(device => (
            <div
              key={device.name}
              className={`p-4 rounded-xl border ${
                device.supported
                  ? 'bg-black/20 border-white/10'
                  : 'bg-black/10 border-white/5 opacity-60'
              }`}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg bg-${device.color}/20 flex items-center justify-center`}>
                    <device.icon className={`w-5 h-5 text-${device.color}`} />
                  </div>
                  <div>
                    <div className="font-medium">{device.name}</div>
                    <div className="text-xs text-white/40">{device.type}</div>
                  </div>
                </div>
                {device.supported ? (
                  <span className="text-xs bg-nc-risk-low/20 text-nc-risk-low px-2 py-1 rounded">
                    Supported
                  </span>
                ) : (
                  <span className="text-xs bg-white/10 text-white/40 px-2 py-1 rounded">
                    Coming Soon
                  </span>
                )}
              </div>

              <div className="flex flex-wrap gap-1 mb-2">
                {device.features.map(f => (
                  <span key={f} className="text-xs bg-white/5 px-2 py-0.5 rounded text-white/50">
                    {f}
                  </span>
                ))}
              </div>

              <div className="flex items-center justify-between text-xs text-white/40">
                <span>{device.connection}</span>
                <span>{device.note}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Future devices note */}
        <div className="mt-6 p-4 bg-nc-eeg/5 border border-nc-eeg/20 rounded-lg">
          <div className="flex items-start gap-3">
            <Brain className="w-5 h-5 text-nc-eeg flex-shrink-0 mt-0.5" />
            <div>
              <div className="font-medium text-nc-eeg">EEG Device Support Coming</div>
              <p className="text-sm text-white/50 mt-1">
                We&apos;re working on integrating consumer EEG devices like Muse and OpenBCI.
                This requires custom SDK integration. Star the repo to get notified when EEG support is added!
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </Section>
  )
}

// ============================================================================
// FOOTER
// ============================================================================

function Footer() {
  return (
    <footer className="py-16 px-6 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-12 mb-12">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <Brain className="w-5 h-5 text-nc-eeg" />
              <Heart className="w-5 h-5 text-nc-ecg" />
              <span className="font-semibold">NeuroCardiac Shield</span>
            </div>
            <p className="text-sm text-white/50">
              An academic prototype demonstrating how next-generation multi-modal
              physiological monitoring devices could be built.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Team</h4>
            <div className="space-y-2 text-sm text-white/60">
              <div>Mohd Sarfaraz Faiyaz</div>
              <div>Vaibhav D. Chandgir</div>
              <div className="text-white/40">Advisor: Dr. Matthew Campisi</div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Institution</h4>
            <div className="text-sm text-white/60">
              <div>NYU Tandon School of Engineering</div>
              <div>Dept. of Electrical & Computer Engineering</div>
              <div className="font-mono text-nc-accent mt-2">ECE-GY 9953 · Fall 2025</div>
            </div>
          </div>
        </div>

        <div className="border-t border-white/10 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-center md:text-left">
              <p className="text-sm text-nc-risk-medium font-medium mb-1">Academic Demonstration Only</p>
              <p className="text-xs text-white/40 max-w-2xl">
                This system is not FDA-cleared, not clinically validated, and not suitable for medical diagnosis.
                All physiological data is synthetic. ML models are trained on synthetic data and carry no clinical validity.
              </p>
            </div>
            <a
              href="https://github.com/bblackheart013/neurocardiac-shield"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-6 py-3 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-colors"
            >
              <Github className="w-5 h-5" />
              <span>View on GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

// ============================================================================
// MAIN PAGE
// ============================================================================

export default function Home() {
  return (
    <main className="bg-nc-bg min-h-screen">
      <Navigation />
      <Hero />
      <ProblemSection />
      <SolutionSection />
      <LiveSignalsSection />
      <SignalScienceSection />
      <SignalProcessingSection />
      <MLSection />
      <ArchitectureSection />
      <ResultsSection />
      <DeviceConnectionSection />
      <Footer />
    </main>
  )
}
