/* ============================================
   Detector de Noticias Falsas — JS
   Anima la barra de confianza al renderizar el resultado.
   ============================================ */

(function () {
    'use strict';

    /**
     * Anima una barra desde 0% hasta su ancho final (data-width).
     */
    function animateBar(fill) {
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

    function init(root) {
        const scope = root || document;
        scope.querySelectorAll('[data-bar-fill]').forEach(animateBar);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => init());
    } else {
        init();
    }

    // Streamlit re-renderiza el DOM en cada interacción; observamos
    // cambios y procesamos las nuevas barras que aparezcan.
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === 1) init(node);
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
})();