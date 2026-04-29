/**
 * SO-101 Robot Controller - Animation Library
 * Version: 1.0.0
 */

const PARC = window.PARC || {};

/* ==========================================================================
   Utility Helpers
   ========================================================================== */
PARC.$ = (sel, ctx = document) => ctx.querySelector(sel);
PARC.$$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
PARC.delay = ms => new Promise(r => setTimeout(r, ms));

PARC.ease = {
    out: 'cubic-bezier(0.16, 1, 0.3, 1)',
    in: 'cubic-bezier(0.4, 0, 1, 1)',
    inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)'
};

/* ==========================================================================
   Transition Manager
   ========================================================================== */
PARC.Transition = {
    fadeIn(el, duration = 300) {
        el.style.transition = `opacity ${duration}ms ${PARC.ease.out}`;
        el.style.opacity = '0';
        el.style.display = 'block';
        requestAnimationFrame(() => {
            el.style.opacity = '1';
        });
        return new Promise(r => setTimeout(r, duration));
    },

    fadeOut(el, duration = 300) {
        return new Promise(r => {
            el.style.transition = `opacity ${duration}ms ${PARC.ease.in}`;
            el.style.opacity = '0';
            setTimeout(() => {
                el.style.display = 'none';
                r();
            }, duration);
        });
    },

    slideIn(el, direction = 'left', duration = 400) {
        const transforms = {
            left: 'translateX(-30px)',
            right: 'translateX(30px)',
            up: 'translateY(-30px)',
            down: 'translateY(30px)'
        };
        el.style.transition = `transform ${duration}ms ${PARC.ease.out}, opacity ${duration}ms ${PARC.ease.out}`;
        el.style.opacity = '0';
        el.style.transform = transforms[direction];
        el.style.display = 'block';
        
        requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translate(0)';
        });
        
        return new Promise(r => setTimeout(r, duration));
    },

    slideOut(el, direction = 'left', duration = 300) {
        const transforms = {
            left: 'translateX(-30px)',
            right: 'translateX(30px)',
            up: 'translateY(-30px)',
            down: 'translateY(30px)'
        };
        el.style.transition = `transform ${duration}ms ${PARC.ease.in}, opacity ${duration}ms ${PARC.ease.in}`;
        
        requestAnimationFrame(() => {
            el.style.opacity = '0';
            el.style.transform = transforms[direction];
        });
        
        return new Promise(r => setTimeout(() => {
            el.style.display = 'none';
            r();
        }, duration));
    }
};

/* ==========================================================================
   Particle System
   ========================================================================== */
PARC.Particles = class Particles {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        this.options = {
            count: options.count || 30,
            colors: options.colors || ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'],
            speed: options.speed || 2,
            size: options.size || 3,
            lifetime: options.lifetime || 2000,
            ...options
        };
        this.particles = [];
        this.running = false;
    }

    start() {
        this.running = true;
        this.createCanvas();
        this.animate();
        
        for (let i = 0; i < this.options.count; i++) {
            setTimeout(() => this.spawn(), i * 100);
        }
    }

    stop() {
        this.running = false;
        if (this.canvas) {
            this.canvas.remove();
            this.canvas = null;
        }
    }

    createCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9999;
            opacity: 0.6;
        `;
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.container.appendChild(this.canvas);
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    spawn() {
        if (!this.running || this.particles.length >= this.options.count) return;
        
        const x = Math.random() * this.canvas.width;
        const y = this.canvas.height + 10;
        const color = this.options.colors[Math.floor(Math.random() * this.options.colors.length)];
        const size = this.options.size * (0.5 + Math.random() * 0.5);
        const speed = this.options.speed * (0.5 + Math.random() * 0.5);
        const angle = -Math.PI / 2 + (Math.random() - 0.5) * 0.5;
        
        this.particles.push({
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            size,
            color,
            alpha: 1,
            decay: 1 / this.options.lifetime * 60
        });
    }

    animate() {
        if (!this.running) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.02; // slight gravity
            p.alpha -= p.decay;
            
            if (p.alpha <= 0) {
                this.particles.splice(i, 1);
                if (this.running) this.spawn();
                continue;
            }
            
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.alpha;
            this.ctx.fill();
        }
        
        this.ctx.globalAlpha = 1;
        requestAnimationFrame(() => this.animate());
    }
};

/* ==========================================================================
   Magnetic Button Effect
   ========================================================================== */
PARC.Magnetic = class Magnetic {
    constructor(el, strength = 0.3) {
        this.el = typeof el === 'string' ? document.querySelector(el) : el;
        this.strength = strength;
        this.enabled = true;
        this.bound = this.onMove.bind(this);
        this.el.addEventListener('mouseenter', () => {
            this.el.style.transition = 'none';
            document.addEventListener('mousemove', this.bound);
        });
        this.el.addEventListener('mouseleave', () => {
            document.removeEventListener('mousemove', this.bound);
            this.el.style.transition = 'transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
            this.el.style.transform = 'translate(0, 0)';
        });
    }

    onMove(e) {
        if (!this.enabled) return;
        const rect = this.el.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = (e.clientX - cx) * this.strength;
        const dy = (e.clientY - cy) * this.strength;
        this.el.style.transform = `translate(${dx}px, ${dy}px)`;
    }
};

/* ==========================================================================
   Tilt Effect (3D card tilt)
   ========================================================================== */
PARC.Tilt = class Tilt {
    constructor(el, intensity = 10) {
        this.el = typeof el === 'string' ? document.querySelector(el) : el;
        this.intensity = intensity;
        this.enabled = true;
        
        this.el.style.transformStyle = 'preserve-3d';
        this.el.addEventListener('mousemove', e => this.onMove(e));
        this.el.addEventListener('mouseleave', () => this.reset());
    }

    onMove(e) {
        if (!this.enabled) return;
        const rect = this.el.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        const dx = (x - cx) / cx;
        const dy = (y - cy) / cy;
        
        this.el.style.transform = `
            perspective(1000px)
            rotateX(${-dy * this.intensity}deg)
            rotateY(${dx * this.intensity}deg)
            translateZ(10px)
        `;
    }

    reset() {
        this.el.style.transition = 'transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)';
        this.el.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateZ(0)';
    }
};

/* ==========================================================================
   Ripple Effect
   ========================================================================== */
PARC.Ripple = {
    apply(el) {
        el.style.position = 'relative';
        el.style.overflow = 'hidden';
        
        el.addEventListener('click', e => {
            const rect = el.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            const ripple = document.createElement('span');
            ripple.style.cssText = `
                position: absolute;
                width: ${size}px;
                height: ${size}px;
                left: ${x}px;
                top: ${y}px;
                background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
                border-radius: 50%;
                transform: scale(0);
                animation: ripple 0.6s ease-out;
                pointer-events: none;
            `;
            
            el.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    }
};

// Add ripple keyframes dynamically
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        to { transform: scale(4); opacity: 0; }
    }
`;
document.head.appendChild(rippleStyle);

/* ==========================================================================
   Typing Effect
   ========================================================================== */
PARC.Typewriter = class Typewriter {
    constructor(el, text, speed = 50) {
        this.el = typeof el === 'string' ? document.querySelector(el) : el;
        this.text = text;
        this.speed = speed;
        this.cursor = '|';
    }

    async start() {
        this.el.textContent = '';
        for (let i = 0; i < this.text.length; i++) {
            this.el.textContent += this.text[i];
            await PARC.delay(this.speed);
        }
        return this;
    }

    async delete(speed = 30) {
        for (let i = this.text.length; i > 0; i--) {
            this.el.textContent = this.text.slice(0, i - 1);
            await PARC.delay(speed);
        }
        return this;
    }
};

/* ==========================================================================
   Counter Animation
   ========================================================================== */
PARC.CountUp = class CountUp {
    constructor(el, end, duration = 1000) {
        this.el = typeof el === 'string' ? document.querySelector(el) : el;
        this.end = parseFloat(end);
        this.duration = duration;
        this.start = 0;
        this.running = false;
    }

    start() {
        this.running = true;
        const startTime = performance.now();
        const animate = (currentTime) => {
            if (!this.running) return;
            
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / this.duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            
            this.el.textContent = Math.round(this.start + (this.end - this.start) * eased);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        requestAnimationFrame(animate);
        return this;
    }

    stop() {
        this.running = false;
    }
};

/* ==========================================================================
   Glitch Effect
   ========================================================================== */
PARC.Glitch = class Glitch {
    constructor(el) {
        this.el = typeof el === 'string' ? document.querySelector(el) : el;
        this.enabled = false;
    }

    start() {
        this.enabled = true;
        this.animate();
    }

    stop() {
        this.enabled = false;
        this.el.style.textShadow = '';
    }

    animate() {
        if (!this.enabled) return;
        
        const effects = [
            `translate(-2px, 2px)`,
            `translate(2px, -2px)`,
            `translate(-2px, -2px)`,
            `translate(2px, 2px)`,
            `translate(0)`
        ];
        
        const colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff'];
        const randomEffect = effects[Math.floor(Math.random() * effects.length)];
        const randomColor = colors[Math.floor(Math.random() * colors.length)];
        
        this.el.style.textShadow = randomEffect;
        
        if (Math.random() > 0.7) {
            this.el.style.textShadow += ` ${randomColor}`;
        }
        
        setTimeout(() => this.animate(), 50 + Math.random() * 100);
    }
};

/* ==========================================================================
   Scroll Animations (Intersection Observer)
   ========================================================================== */
PARC.ScrollAnim = {
    observer: null,
    
    init() {
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    if (entry.target.dataset.scrollOnce) {
                        this.observer.unobserve(entry.target);
                    }
                }
            });
        }, { threshold: 0.1 });
    },

    observe(el) {
        el.classList.add('scroll-animate');
        el.style.opacity = '0';
        this.observer.observe(el);
    },

    observeAll(sel) {
        document.querySelectorAll(sel).forEach(el => this.observe(el));
    }
};

/* ==========================================================================
   Cursor Trail
   ========================================================================== */
PARC.CursorTrail = class CursorTrail {
    constructor(options = {}) {
        this.options = {
            color: options.color || '#3b82f6',
            size: options.size || 4,
            length: options.length || 20,
            speed: options.speed || 0.5,
            ...options
        };
        this.points = [];
        this.enabled = false;
    }

    enable() {
        this.enabled = true;
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 9998;
        `;
        document.body.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        this.resize();
        window.addEventListener('resize', () => this.resize());
        document.addEventListener('mousemove', e => this.onMove(e));
        this.animate();
    }

    disable() {
        this.enabled = false;
        if (this.canvas) {
            this.canvas.remove();
            this.canvas = null;
        }
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    onMove(e) {
        this.points.push({ x: e.clientX, y: e.clientY });
        if (this.points.length > this.options.length) {
            this.points.shift();
        }
    }

    animate() {
        if (!this.enabled || !this.canvas) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            const alpha = i / this.points.length;
            const size = this.options.size * alpha;
            
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
            this.ctx.fillStyle = this.options.color;
            this.ctx.globalAlpha = alpha * 0.5;
            this.ctx.fill();
        }
        
        this.ctx.globalAlpha = 1;
        requestAnimationFrame(() => this.animate());
    }
};

/* ==========================================================================
   Notification Toast
   ========================================================================== */
PARC.Toast = {
    container: null,
    
    init() {
        this.container = document.createElement('div');
        this.container.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 10000;
            display: flex;
            flex-direction: column-reverse;
            gap: 8px;
        `;
        document.body.appendChild(this.container);
    },

    show(message, type = 'info', duration = 3000) {
        if (!this.container) this.init();
        
        const colors = {
            info: ['#3b82f6', '#111624'],
            success: ['#10b981', '#111624'],
            warning: ['#f59e0b', '#111624'],
            error: ['#ef4444', '#111624']
        };
        
        const [color, bg] = colors[type] || colors.info;
        
        const toast = document.createElement('div');
        toast.style.cssText = `
            background: ${bg};
            border: 1px solid ${color};
            border-left: 3px solid ${color};
            border-radius: 8px;
            padding: 12px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #e2e8f0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            transform: translateX(120%);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            max-width: 300px;
        `;
        toast.textContent = message;
        
        this.container.appendChild(toast);
        requestAnimationFrame(() => {
            toast.style.transform = 'translateX(0)';
        });
        
        setTimeout(() => {
            toast.style.transform = 'translateX(120%)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
};

/* ==========================================================================
   Smooth Scroll
   ========================================================================== */
PARC.SmoothScroll = {
    scrollTo(target, offset = 0) {
        const el = typeof target === 'string' ? document.querySelector(target) : target;
        if (!el) return;
        
        const top = el.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({
            top,
            behavior: 'smooth'
        });
    },

    init() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', e => {
                e.preventDefault();
                const target = anchor.getAttribute('href');
                this.scrollTo(target);
            });
        });
    }
};

/* ==========================================================================
   Init on DOM Ready
   ========================================================================== */
PARC.initAnimations = function() {
    PARC.ScrollAnim.init();
    PARC.SmoothScroll.init();
    
    // Apply ripple to all buttons
    document.querySelectorAll('.btn').forEach(btn => PARC.Ripple.apply(btn));
    
    console.log('[PARC] Animations initialized');
};

// Auto-init
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', PARC.initAnimations);
} else {
    PARC.initAnimations();
}

window.PARC = PARC;