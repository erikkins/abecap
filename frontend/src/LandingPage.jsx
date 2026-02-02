import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  TrendingUp, Zap, Shield, BarChart3, Bell, DollarSign,
  CheckCircle, ArrowRight, ChevronDown, ChevronUp, Star, Users, Target
} from 'lucide-react';
import { useAuth } from './contexts/AuthContext';
import LoginModal from './components/LoginModal';

// ============================================================================
// Landing Page Components
// ============================================================================

const HeroSection = ({ onGetStarted }) => (
  <section className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-700 text-white">
    {/* Animated background elements */}
    <div className="absolute inset-0 overflow-hidden">
      <div className="absolute -top-40 -right-40 w-80 h-80 bg-white/10 rounded-full blur-3xl"></div>
      <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl"></div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-blue-400/10 rounded-full blur-3xl"></div>
      {/* Floating chart lines */}
      <svg className="absolute inset-0 w-full h-full opacity-10" viewBox="0 0 1200 600">
        <path d="M0,400 Q200,350 400,380 T800,300 T1200,350" fill="none" stroke="white" strokeWidth="2"/>
        <path d="M0,450 Q300,400 600,420 T1200,380" fill="none" stroke="white" strokeWidth="1.5"/>
      </svg>
    </div>

    <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-32">
      <div className="text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-sm font-medium mb-8">
          <Zap className="w-4 h-4 text-yellow-300" />
          <span>15 Years of Algorithm Development</span>
          <span className="px-2 py-0.5 bg-emerald-500 rounded-full text-xs">263% Backtest Return</span>
        </div>

        {/* Main headline */}
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
          Beat the Market with
          <span className="block bg-gradient-to-r from-yellow-200 via-yellow-300 to-orange-300 bg-clip-text text-transparent pb-2">
            AI-Powered Trading Signals
          </span>
        </h1>

        {/* Subheadline */}
        <p className="max-w-2xl mx-auto text-lg sm:text-xl text-blue-100 mb-10">
          15 years of continuous refinement. Our algorithms have evolved through every market cycle
          since 2011 - always adapting, always improving, always delivering positive returns.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
          <button
            onClick={onGetStarted}
            className="group flex items-center gap-2 px-8 py-4 bg-white text-indigo-600 font-semibold rounded-xl shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all"
          >
            Start Free Trial
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
          <a
            href="#how-it-works"
            className="flex items-center gap-2 px-8 py-4 text-white/90 font-medium hover:text-white transition-colors"
          >
            See How It Works
            <ChevronDown className="w-5 h-5" />
          </a>
        </div>

        {/* Social proof */}
        <div className="flex flex-wrap justify-center items-center gap-8 text-sm text-blue-200">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5" />
            <span><strong className="text-white">2,500+</strong> Active Traders</span>
          </div>
          <div className="flex items-center gap-2">
            <Star className="w-5 h-5 text-yellow-300" />
            <span><strong className="text-white">15</strong> Years of Development</span>
          </div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-300" />
            <span><strong className="text-white">263%</strong> Total Return (15yr)</span>
          </div>
        </div>
      </div>
    </div>

    {/* Wave divider */}
    <div className="absolute bottom-0 left-0 right-0">
      <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M0,64 C480,150 960,-20 1440,64 L1440,120 L0,120 Z" fill="white"/>
      </svg>
    </div>
  </section>
);

const FeatureCard = ({ icon: Icon, title, description, color }) => (
  <div className="group relative bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all border border-gray-100">
    <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl ${color} mb-6`}>
      <Icon className="w-7 h-7 text-white" />
    </div>
    <h3 className="text-xl font-semibold text-gray-900 mb-3">{title}</h3>
    <p className="text-gray-600 leading-relaxed">{description}</p>
  </div>
);

const FeaturesSection = () => (
  <section className="py-20 bg-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
          Everything You Need to Trade Smarter
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Our platform combines institutional-grade analysis with a simple interface anyone can use.
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
        <FeatureCard
          icon={Zap}
          title="Real-Time Signals"
          description="Get instant buy alerts when our algorithms identify high-conviction opportunities. Never miss a winning trade again."
          color="bg-gradient-to-br from-yellow-400 to-orange-500"
        />
        <FeatureCard
          icon={Shield}
          title="Built-In Risk Management"
          description="Every signal includes stop-loss and profit targets. Protect your capital with proven exit rules."
          color="bg-gradient-to-br from-blue-500 to-indigo-600"
        />
        <FeatureCard
          icon={BarChart3}
          title="Advanced Analytics"
          description="Track your performance with detailed charts, P&L tracking, and portfolio analytics."
          color="bg-gradient-to-br from-purple-500 to-pink-500"
        />
        <FeatureCard
          icon={Bell}
          title="Daily Email Digest"
          description="Receive a beautiful summary of top signals every evening, right to your inbox."
          color="bg-gradient-to-br from-emerald-500 to-teal-600"
        />
        <FeatureCard
          icon={Target}
          title="52-Week High Scanner"
          description="Identify breakout candidates near their all-time highs with strong momentum."
          color="bg-gradient-to-br from-red-500 to-rose-600"
        />
        <FeatureCard
          icon={DollarSign}
          title="Position Sizing"
          description="Automatic position size calculations based on your portfolio and risk tolerance."
          color="bg-gradient-to-br from-green-500 to-emerald-600"
        />
      </div>
    </div>
  </section>
);

const HowItWorksSection = () => (
  <section id="how-it-works" className="py-20 bg-gradient-to-b from-gray-50 to-white">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
          How RigaCap Works
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          From signal to profit in 4 simple steps
        </p>
      </div>

      <div className="grid md:grid-cols-4 gap-8">
        {[
          { step: 1, title: 'Scan', desc: 'Our algorithms analyze 4,500+ stocks daily, ranking by momentum, quality, and market conditions' },
          { step: 2, title: 'Signal', desc: 'When criteria align, you get actionable alerts with entry, stop, and target levels' },
          { step: 3, title: 'Buy', desc: 'Execute with confidence - every signal includes built-in risk management' },
          { step: 4, title: 'Profit', desc: 'Trailing stops protect gains while letting winners run' },
        ].map(({ step, title, desc }) => (
          <div key={step} className="relative text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-600 text-white text-2xl font-bold mb-4">
              {step}
            </div>
            {step < 4 && (
              <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-0.5 bg-indigo-200"></div>
            )}
            <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
            <p className="text-gray-600">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const StatsSection = () => (
  <section className="py-16 bg-indigo-600">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-8 text-center text-white">
        {[
          { value: '263%', label: 'Total Return' },
          { value: '9%', label: 'Annual Return' },
          { value: '1.15', label: 'Sharpe Ratio' },
          { value: '49%', label: 'Win Rate' },
          { value: '-14%', label: 'Max Drawdown' },
        ].map(({ value, label }) => (
          <div key={label}>
            <div className="text-4xl lg:text-5xl font-bold mb-2">{value}</div>
            <div className="text-indigo-200">{label}</div>
          </div>
        ))}
      </div>
      <p className="text-indigo-200 text-sm mt-6 text-center max-w-2xl mx-auto">
        15 years of continuous refinement. Our algorithms have evolved through the 2011 recovery,
        2018 correction, COVID crash, and 2022 bear market - always adapting to deliver positive returns.
        Results are backtested (2011-2026).
      </p>
    </div>
  </section>
);

const PricingSection = ({ onGetStarted }) => {
  const features = [
    'Unlimited real-time signals',
    'Daily email digest',
    'Stop-loss & profit targets',
    'Portfolio tracking',
    'Performance analytics',
    'Mobile-friendly dashboard',
    '4,500+ stocks scanned daily',
    'Sector & company insights',
  ];

  return (
    <section id="pricing" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Choose monthly flexibility or save with annual billing. Cancel anytime.
          </p>
        </div>

        <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8">
          {/* Monthly Plan */}
          <div className="relative bg-white rounded-3xl shadow-xl border border-gray-200 overflow-hidden">
            <div className="p-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Monthly</h3>
              <p className="text-gray-600 mb-6">Pay as you go</p>

              <div className="flex items-baseline gap-2 mb-8">
                <span className="text-5xl font-bold text-gray-900">$20</span>
                <span className="text-gray-500">/month</span>
              </div>

              <ul className="space-y-3 mb-8">
                {features.map((feature) => (
                  <li key={feature} className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-500 flex-shrink-0" />
                    <span className="text-gray-700 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => onGetStarted('monthly')}
                className="w-full py-4 bg-gray-100 text-gray-900 font-semibold rounded-xl hover:bg-gray-200 transition-colors"
              >
                Start 7-Day Free Trial
              </button>
              <p className="text-center text-sm text-gray-500 mt-4">
                No credit card required
              </p>
            </div>
          </div>

          {/* Annual Plan */}
          <div className="relative bg-white rounded-3xl shadow-2xl border-2 border-indigo-500 overflow-hidden">
            {/* Best Value Badge */}
            <div className="absolute top-0 right-0 bg-indigo-500 text-white px-4 py-1 text-sm font-semibold rounded-bl-xl">
              BEST VALUE
            </div>
            <div className="p-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Annual</h3>
              <p className="text-gray-600 mb-6">Save with yearly billing</p>

              <div className="flex items-baseline gap-2 mb-2">
                <span className="text-5xl font-bold text-gray-900">$200</span>
                <span className="text-gray-500">/year</span>
              </div>
              <p className="text-emerald-600 font-medium mb-6">
                Save $40 — 2 months free!
              </p>

              <ul className="space-y-3 mb-8">
                {features.map((feature) => (
                  <li key={feature} className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-emerald-500 flex-shrink-0" />
                    <span className="text-gray-700 text-sm">{feature}</span>
                  </li>
                ))}
              </ul>

              <button
                onClick={() => onGetStarted('annual')}
                className="w-full py-4 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 transition-colors"
              >
                Start 7-Day Free Trial
              </button>
              <p className="text-center text-sm text-gray-500 mt-4">
                No credit card required
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const TestimonialCard = ({ quote, author, role, avatar }) => (
  <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-100">
    <div className="flex gap-1 mb-4">
      {[1,2,3,4,5].map(i => (
        <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
      ))}
    </div>
    <p className="text-gray-700 mb-6 leading-relaxed">"{quote}"</p>
    <div className="flex items-center gap-4">
      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 flex items-center justify-center text-white font-bold">
        {avatar}
      </div>
      <div>
        <div className="font-semibold text-gray-900">{author}</div>
        <div className="text-sm text-gray-500">{role}</div>
      </div>
    </div>
  </div>
);

const TestimonialsSection = () => (
  <section className="py-20 bg-gray-50">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
          Loved by Traders Everywhere
        </h2>
        <p className="text-lg text-gray-600">
          See what our members are saying
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-8">
        <TestimonialCard
          quote="RigaCap's signals have completely changed how I trade. The DWAP strategy is brilliant - I've made more in 3 months than the entire previous year."
          author="Michael R."
          role="Day Trader, Texas"
          avatar="MR"
        />
        <TestimonialCard
          quote="Finally, a trading tool that doesn't overwhelm me with complexity. Simple signals, clear targets. My portfolio is up 45% since I started."
          author="Sarah K."
          role="Part-time Investor, NYC"
          avatar="SK"
        />
        <TestimonialCard
          quote="The daily email digest is worth the subscription alone. I check it every evening and plan my next day's trades. Absolutely essential."
          author="James L."
          role="Swing Trader, Chicago"
          avatar="JL"
        />
      </div>
    </div>
  </section>
);

const FAQItem = ({ question, answer }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-gray-200">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-6 text-left"
      >
        <span className="text-lg font-medium text-gray-900">{question}</span>
        {open ? (
          <ChevronUp className="w-5 h-5 text-gray-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-500" />
        )}
      </button>
      {open && (
        <div className="pb-6 text-gray-600 leading-relaxed">
          {answer}
        </div>
      )}
    </div>
  );
};

const FAQSection = () => (
  <section className="py-20 bg-white">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="text-center mb-16">
        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">
          Frequently Asked Questions
        </h2>
      </div>

      <div className="divide-y divide-gray-200">
        <FAQItem
          question="How does RigaCap generate signals?"
          answer="Our algorithms have been refined over 15 years of development. We analyze momentum, volatility, market regime, and dozens of other factors. When conditions are unfavorable, we recommend staying in cash. When opportunity strikes, we identify the highest-conviction picks. The strategy continuously evolves based on what's working."
        />
        <FAQItem
          question="How many signals do you generate per day?"
          answer="On average, we identify 3-8 quality signals per week from our universe of 4,500+ stocks. We prioritize quality over quantity - our focused approach targets only the best risk-adjusted opportunities."
        />
        <FAQItem
          question="What's your track record?"
          answer="Our backtested strategy returned 263% total over 15 years (2011-2026), averaging 9% annually with a 49% win rate and -14% max drawdown. We've navigated the 2011 recovery, 2018 correction, COVID crash, and 2022 bear market - continuously adapting through each cycle."
        />
        <FAQItem
          question="Can I cancel anytime?"
          answer="Absolutely! There are no contracts or commitments. You can cancel your subscription anytime from your account settings, and you'll retain access until the end of your billing period."
        />
        <FAQItem
          question="Do you provide financial advice?"
          answer="No, RigaCap provides algorithmic signals and educational information only. We are not financial advisors. Always do your own research and consider consulting a licensed professional before making investment decisions."
        />
      </div>
    </div>
  </section>
);

const CTASection = ({ onGetStarted }) => (
  <section className="py-20 bg-gradient-to-br from-indigo-600 to-purple-700 text-white">
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
      <h2 className="text-3xl sm:text-4xl font-bold mb-6">
        Ready to Start Winning?
      </h2>
      <p className="text-xl text-indigo-100 mb-10 max-w-2xl mx-auto">
        Join thousands of traders using AI-powered signals to find winning opportunities.
        Your first week is free - no credit card required.
      </p>
      <button
        onClick={onGetStarted}
        className="group inline-flex items-center gap-2 px-10 py-5 bg-white text-indigo-600 font-bold text-lg rounded-xl shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all"
      >
        Start Your Free Trial
        <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
      </button>
    </div>
  </section>
);

const Footer = () => (
  <footer className="bg-gray-900 text-gray-400 py-12">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col md:flex-row justify-between items-center gap-6">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-8 h-8 text-indigo-400" />
          <span className="text-xl font-bold text-white">RigaCap</span>
        </div>
        <div className="flex gap-8 text-sm">
          <a href="#" className="hover:text-white transition-colors">Terms</a>
          <a href="#" className="hover:text-white transition-colors">Privacy</a>
          <a href="#" className="hover:text-white transition-colors">Contact</a>
        </div>
        <div className="text-sm">
          &copy; {new Date().getFullYear()} RigaCap. All rights reserved.
        </div>
      </div>
      <div className="mt-8 pt-8 border-t border-gray-800 text-center text-sm">
        <p>
          Trading involves risk. Past performance does not guarantee future results.
          RigaCap provides algorithmic signals for educational purposes only and is not a registered investment advisor.
        </p>
      </div>
    </div>
  </footer>
);

const Navbar = ({ onGetStarted }) => (
  <nav className="absolute top-0 left-0 right-0 z-50">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-8 h-8 text-white" />
          <span className="text-xl font-bold text-white">RigaCap</span>
        </div>
        <div className="hidden md:flex items-center gap-8 text-white/80">
          <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
          <a href="#pricing" className="hover:text-white transition-colors">Pricing</a>
          <button
            onClick={onGetStarted}
            className="px-5 py-2 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-colors"
          >
            Sign In
          </button>
        </div>
      </div>
    </div>
  </nav>
);

// ============================================================================
// Main Landing Page Component
// ============================================================================

// Helper to check if user has visited before (cookie-based)
const hasVisitedBefore = () => {
  return document.cookie.includes('rigacap_visited=true');
};

const setVisitedCookie = () => {
  // Set cookie for 1 year
  const expires = new Date();
  expires.setFullYear(expires.getFullYear() + 1);
  document.cookie = `rigacap_visited=true; expires=${expires.toUTCString()}; path=/; SameSite=Lax`;
};

export default function LandingPage() {
  const navigate = useNavigate();
  const { isAuthenticated, loading } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState('monthly');
  const [isReturningVisitor, setIsReturningVisitor] = useState(false);

  // Check if returning visitor on mount
  useEffect(() => {
    setIsReturningVisitor(hasVisitedBefore());
    setVisitedCookie(); // Mark as visited
  }, []);

  // Redirect if already authenticated (after loading is complete)
  useEffect(() => {
    if (!loading && isAuthenticated) {
      console.log('LandingPage: User authenticated, navigating to /app');
      navigate('/app', { replace: true });
    }
  }, [isAuthenticated, loading, navigate]);

  const handleGetStarted = (plan = 'monthly') => {
    if (isAuthenticated) {
      navigate('/app', { replace: true });
    } else {
      setSelectedPlan(plan);
      setShowLoginModal(true);
    }
  };

  const handleLoginSuccess = () => {
    console.log('LandingPage: handleLoginSuccess called');
    // Close modal first
    setShowLoginModal(false);
    // Navigate immediately - auth state is already updated in context
    navigate('/app', { replace: true });
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar onGetStarted={handleGetStarted} />
      <HeroSection onGetStarted={handleGetStarted} />
      <FeaturesSection />
      <HowItWorksSection />
      <StatsSection />
      <PricingSection onGetStarted={handleGetStarted} />
      <TestimonialsSection />
      <FAQSection />
      <CTASection onGetStarted={handleGetStarted} />
      <Footer />

      {/* Login Modal */}
      {showLoginModal && (
        <LoginModal
          onClose={() => setShowLoginModal(false)}
          onSuccess={handleLoginSuccess}
          initialMode={isReturningVisitor ? 'login' : 'register'}
          selectedPlan={selectedPlan}
        />
      )}
    </div>
  );
}
