/* ============================================
   Loan Approval Predictor — Dashboard JS
   - Anima contadores numéricos en metric-cards
   - Formatea valores monetarios con separadores
   - Anima la barra del CIBIL gauge al renderizar
   ============================================ */

(function () {
    'use strict';

    /**
     * Formatea un número como USD con separadores de miles.
     * 5000000 -> "$5,000,000"
     */
    function formatCurrency(value) {
        return '$' + Math.round(value).toLocaleString('en-US');
    }

    /**
     * Anima un número desde 0 hasta su valor objetivo.
     * Soporta el atributo data-format="currency|percent|ratio|integer".
     */
    function animateCounter(el) {
        if (el.dataset.animated === 'true') return;
        el.dataset.animated = 'true';

        const target = parseFloat(el.dataset.target);
        if (isNaN(target)) return;

        const format = el.dataset.format || 'integer';
        const duration = 900;
        const start = performance.now();

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;

            switch (format) {
                case 'currency':
                    el.textContent = formatCurrency(current);
                    break;
                case 'percent':
                    el.textContent = current.toFixed(1) + '%';
                    break;
                case 'ratio':
                    el.textContent = current.toFixed(2) + 'x';
                    break;
                case 'integer':
                default:
                    el.textContent = Math.round(current).toLocaleString('en-US');
                    break;
            }

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);
    }

    /**
     * Anima la barra del CIBIL gauge desde 0 hasta el ancho final.
     */
    function animateGauge(fill) {
        if (fill.dataset.animated === 'true') return;
        fill.dataset.animated = 'true';

        const targetWidth = fill.dataset.width;
        if (!targetWidth) return;

        fill.style.width = '0%';
        // forzar reflow para que la transición CSS tome efecto
        // eslint-disable-next-line no-unused-expressions
        fill.offsetWidth;
        requestAnimationFrame(() => {
            fill.style.width = targetWidth + '%';
        });
    }

    /**
     * Busca todos los elementos animables y los procesa.
     * Se llama tanto en load inicial como cuando Streamlit
     * re-renderiza (vía MutationObserver del documento).
     */
    function initAnimations(root) {
        const scope = root || document;
        scope.querySelectorAll('[data-counter]').forEach(animateCounter);
        scope.querySelectorAll('[data-gauge-fill]').forEach(animateGauge);
    }

    // Inicial
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => initAnimations());
    } else {
        initAnimations();
    }

    // Streamlit re-renderiza el DOM en cada interacción, así que
    // observamos cambios y re-procesamos nuevos elementos.
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === 1) {
                    initAnimations(node);
                }
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
})();