#include"../include/draw.h"
#include"internal/blend_utils.h"

using namespace pa2d::utils;
using namespace pa2d::utils::simd;

namespace pa2d {
    // 预计算的边数据结构，对齐以优化加载
    struct EdgeParams {
        float ax, ay;
        float dx, dy;
        float invLenSq; // 1.0 / (dx*dx + dy*dy)

        // 射线法相关参数
        float yMin, yMax;
        float xOfYMin;    // yMin 对应的 x 坐标
        float invSlope;   // (x2 - x1) / (y2 - y1)，用于快速计算交点 X
        bool  isHorizontal; // 是否水平线
    };


    void polygon(
        pa2d::Buffer& buffer,
        const std::vector<pa2d::Point>& vertices,
        const pa2d::Color& fillColor,
        const pa2d::Color& strokeColor,
        float strokeWidth
    ) {
        if (!buffer.isValid() || vertices.size() < 3) return;

        // 1. 颜色与不透明度预处理
        const float finalFillOpacity = fillColor.a * (1.0f / 255.0f);
        const float finalStrokeOpacity = strokeColor.a * (1.0f / 255.0f);
        const bool drawFill = (finalFillOpacity > 0.0f);
        const bool drawStroke = (finalStrokeOpacity > 0.0f) && (strokeWidth > 0.0f);
        if (!drawFill && !drawStroke) return;

        const bool mode_stroke_over_fill = drawStroke && drawFill;
        const bool mode_only_stroke = drawStroke && !drawFill;

        // 2. 几何参数与包围盒
        const float halfStrokeWidth = strokeWidth == 0 ? 0 : (strokeWidth + 1.0) * 0.5f;
        const float maxExt = halfStrokeWidth + 1.0f;

        pa2d::Point minPt = vertices[0];
        pa2d::Point maxPt = vertices[0];
        size_t n = vertices.size();

        // 3. 预计算所有边参数 (关键优化)
        std::vector<EdgeParams> edges;
        edges.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            const pa2d::Point& p1 = vertices[i];
            const pa2d::Point& p2 = vertices[(i + 1) % n];

            minPt.x = std::min(minPt.x, p1.x); minPt.y = std::min(minPt.y, p1.y);
            maxPt.x = std::max(maxPt.x, p1.x); maxPt.y = std::max(maxPt.y, p1.y);

            EdgeParams e;
            e.ax = p1.x; e.ay = p1.y;
            e.dx = p2.x - p1.x;
            e.dy = p2.y - p1.y;
            float lenSq = e.dx * e.dx + e.dy * e.dy;
            e.invLenSq = (lenSq > GEOMETRY_EPSILON) ? (1.0f / lenSq) : 0.0f;

            // 射线法预处理
            e.isHorizontal = (std::abs(e.dy) < GEOMETRY_EPSILON);
            if (p1.y < p2.y) {
                e.yMin = p1.y; e.yMax = p2.y; e.xOfYMin = p1.x;
            }
            else {
                e.yMin = p2.y; e.yMax = p1.y; e.xOfYMin = p2.x;
            }
            e.invSlope = e.isHorizontal ? 0.0f : (e.dx / e.dy); // 水平线不会触发相交逻辑，invSlope 设为0安全

            edges.push_back(e);
        }

        int minX = std::max(0, static_cast<int>(std::floor(minPt.x - maxExt)));
        int maxX = std::min(buffer.width - 1, static_cast<int>(std::ceil(maxPt.x + maxExt)));
        int minY = std::max(0, static_cast<int>(std::floor(minPt.y - maxExt)));
        int maxY = std::min(buffer.height - 1, static_cast<int>(std::ceil(maxPt.y + maxExt)));

        if (minX > maxX || minY > maxY) return;

        // 4. SIMD 常量
        const __m256 halfStrokeWidth_v = _mm256_set1_ps(halfStrokeWidth);
        const __m256 v_FLT_MAX = _mm256_set1_ps(FLT_MAX);
        // 符号位掩码: 0x80000000
        const __m256 sign_bit_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        // 颜色常量
        const __m256 fillR_v = _mm256_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m256 fillG_v = _mm256_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m256 fillB_v = _mm256_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m256 fillA_v = _mm256_set1_ps(finalFillOpacity);

        const __m256 strokeR_v = _mm256_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m256 strokeG_v = _mm256_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m256 strokeB_v = _mm256_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m256 strokeA_v = _mm256_set1_ps(finalStrokeOpacity);

        // SSE 常量
        const __m128 halfStrokeWidth_sse = _mm_set1_ps(halfStrokeWidth);
        const __m128 v_FLT_MAX_sse = _mm_set1_ps(FLT_MAX);
        const __m128 sign_bit_mask_sse = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

        const __m128 fillR_sse = _mm_set1_ps(fillColor.r * (1.0f / 255.0f));
        const __m128 fillG_sse = _mm_set1_ps(fillColor.g * (1.0f / 255.0f));
        const __m128 fillB_sse = _mm_set1_ps(fillColor.b * (1.0f / 255.0f));
        const __m128 fillA_sse = _mm_set1_ps(finalFillOpacity);

        const __m128 strokeR_sse = _mm_set1_ps(strokeColor.r * (1.0f / 255.0f));
        const __m128 strokeG_sse = _mm_set1_ps(strokeColor.g * (1.0f / 255.0f));
        const __m128 strokeB_sse = _mm_set1_ps(strokeColor.b * (1.0f / 255.0f));
        const __m128 strokeA_sse = _mm_set1_ps(finalStrokeOpacity);

        // 5. 渲染循环
        for (int py = minY; py <= maxY; ++py) {
            const float fy = static_cast<float>(py) + 0.5f;
            pa2d::Color* row = &buffer.at(0, py);

            const __m256 py_v = _mm256_set1_ps(fy);
            const __m128 py_v_sse = _mm_set1_ps(fy);

            int px = minX;

            // --- AVX2 Loop (8 pixels) ---
            for (; px <= maxX - 7; px += 8) {
                __m256i xBase = _mm256_setr_epi32(px, px + 1, px + 2, px + 3, px + 4, px + 5, px + 6, px + 7);
                __m256 px_v = _mm256_add_ps(_mm256_cvtepi32_ps(xBase), HALF_PIXEL_256);

                __m256 min_dist_sq = v_FLT_MAX;
                // 用于累积奇偶规则的掩码 (0 = 外部, All 1s = 内部)
                // 初始设为 0 (外部)
                __m256 inside_mask_acc = ZERO_256;

                // 遍历所有边 (同时计算距离和符号，提高缓存利用率)
                for (const auto& edge : edges) {
                    // 1. Distance Squared to Segment
                    __m256 v_ax = _mm256_set1_ps(edge.ax);
                    __m256 v_ay = _mm256_set1_ps(edge.ay);
                    __m256 v_dx = _mm256_set1_ps(edge.dx);
                    __m256 v_dy = _mm256_set1_ps(edge.dy);
                    __m256 v_invLenSq = _mm256_set1_ps(edge.invLenSq);

                    __m256 p_ax = _mm256_sub_ps(px_v, v_ax);
                    __m256 p_ay = _mm256_sub_ps(py_v, v_ay);
                    __m256 dot = _mm256_add_ps(_mm256_mul_ps(p_ax, v_dx), _mm256_mul_ps(p_ay, v_dy));
                    __m256 t = _mm256_mul_ps(dot, v_invLenSq);
                    t = _mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t));

                    __m256 closestX = _mm256_add_ps(v_ax, _mm256_mul_ps(t, v_dx));
                    __m256 closestY = _mm256_add_ps(v_ay, _mm256_mul_ps(t, v_dy));
                    __m256 diffX = _mm256_sub_ps(px_v, closestX);
                    __m256 diffY = _mm256_sub_ps(py_v, closestY);
                    __m256 distSq = _mm256_add_ps(_mm256_mul_ps(diffX, diffX), _mm256_mul_ps(diffY, diffY));
                    min_dist_sq = _mm256_min_ps(min_dist_sq, distSq);

                    // 2. Ray Casting Inside/Outside Test (Vectorized)
                    // 只有当边跨越当前扫描线 y 时才计算
                    // 标量判断：这一行对于这一条边是否"感兴趣"
                    if (edge.yMin <= fy && edge.yMax > fy) {
                        // 计算交点的 X 坐标 (对于这 8 个像素，y 相同，所以 X 交点相同)
                        // X = x_at_ymin + (fy - ymin) * invSlope
                        float intersectX = edge.xOfYMin + (fy - edge.yMin) * edge.invSlope;

                        // 如果 intersectX > px，则交叉数 +1 (奇偶规则翻转)
                        __m256 v_intersect = _mm256_set1_ps(intersectX);
                        // 比较: intersectX > px
                        __m256 cross_mask = _mm256_cmp_ps(v_intersect, px_v, _CMP_GT_OQ);
                        // 使用 XOR 翻转状态
                        inside_mask_acc = _mm256_xor_ps(inside_mask_acc, cross_mask);
                    }
                }

                __m256 dist_unsigned = _mm256_sqrt_ps(min_dist_sq);

                // 应用符号：如果 mask 为 true (NaN/All 1s)，则是内部 (-)，否则是外部 (+)
                // inside_mask_acc 为 1s (True) 时代表内部
                // 我们需要: Inside -> Negative SDF, Outside -> Positive SDF
                // 可以使用 bitwise OR with sign bit mask 如果是内部
                // 或者更简单：blend 
                // Mask True (Inside) -> -dist, False (Outside) -> dist
                __m256 neg_dist = _mm256_sub_ps(ZERO_256, dist_unsigned);
                __m256 sdf = _mm256_blendv_ps(dist_unsigned, neg_dist, inside_mask_acc);

                // --- Alpha & Blending (Standard) ---
                __m256 effectiveFillAlpha = ZERO_256;
                if (drawFill) {
                    __m256 t_fill = _mm256_div_ps(_mm256_sub_ps(sdf, ANTIALIAS_RANGE_256), ANTIALIAS_RANGE_256);
                    __m256 fillAlpha_raw = _mm256_sub_ps(ONE_256, t_fill);
                    effectiveFillAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, fillAlpha_raw)), fillA_v);
                }

                __m256 effectiveStrokeAlpha = ZERO_256;
                if (drawStroke) {
                    __m256 halfWidthMinusDist = _mm256_sub_ps(halfStrokeWidth_v, dist_unsigned);
                    __m256 t_stroke = _mm256_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_256);
                    effectiveStrokeAlpha = _mm256_mul_ps(_mm256_max_ps(ZERO_256, _mm256_min_ps(ONE_256, t_stroke)), strokeA_v);
                }

                __m256 finalAlpha, finalR, finalG, finalB;

                if (mode_stroke_over_fill) {
                    __m256 oneMinusStrk = _mm256_sub_ps(ONE_256, effectiveStrokeAlpha);
                    __m256 effFillMod = _mm256_mul_ps(effectiveFillAlpha, oneMinusStrk);
                    finalAlpha = _mm256_add_ps(effectiveStrokeAlpha, effFillMod);

                    finalR = _mm256_add_ps(_mm256_mul_ps(strokeR_v, effectiveStrokeAlpha), _mm256_mul_ps(fillR_v, effFillMod));
                    finalG = _mm256_add_ps(_mm256_mul_ps(strokeG_v, effectiveStrokeAlpha), _mm256_mul_ps(fillG_v, effFillMod));
                    finalB = _mm256_add_ps(_mm256_mul_ps(strokeB_v, effectiveStrokeAlpha), _mm256_mul_ps(fillB_v, effFillMod));

                    __m256 invFinalAlpha = _mm256_div_ps(ONE_256, finalAlpha);
                    __m256 zero_mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_EQ_OQ);
                    invFinalAlpha = _mm256_andnot_ps(zero_mask, invFinalAlpha);
                    finalR = _mm256_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm256_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm256_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_v; finalG = strokeG_v; finalB = strokeB_v;
                }
                else {
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_v; finalG = fillG_v; finalB = fillB_v;
                }

                __m256 mask = _mm256_cmp_ps(finalAlpha, ZERO_256, _CMP_GT_OQ);
                if (_mm256_testz_ps(mask, mask)) continue;

                __m256i dest = _mm256_loadu_si256((__m256i*) & row[px]);
                __m256i rgba = blend_pixels_avx(finalAlpha, dest, finalR, finalG, finalB);
                rgba = _mm256_blendv_epi8(dest, rgba, _mm256_castps_si256(mask));
                _mm256_storeu_si256((__m256i*) & row[px], rgba);
            }

            // --- SSE Loop (4 pixels) ---
            for (; px <= maxX - 3; px += 4) {
                __m128i xBase = _mm_setr_epi32(px, px + 1, px + 2, px + 3);
                __m128 px_v = _mm_add_ps(_mm_cvtepi32_ps(xBase), HALF_PIXEL_128);

                __m128 min_dist_sq = v_FLT_MAX_sse;
                __m128 inside_mask_acc = ZERO_128;

                for (const auto& edge : edges) {
                    __m128 v_ax = _mm_set1_ps(edge.ax);
                    __m128 v_ay = _mm_set1_ps(edge.ay);
                    __m128 v_dx = _mm_set1_ps(edge.dx);
                    __m128 v_dy = _mm_set1_ps(edge.dy);
                    __m128 v_invLenSq = _mm_set1_ps(edge.invLenSq);

                    __m128 p_ax = _mm_sub_ps(px_v, v_ax);
                    __m128 p_ay = _mm_sub_ps(py_v_sse, v_ay);
                    __m128 dot = _mm_add_ps(_mm_mul_ps(p_ax, v_dx), _mm_mul_ps(p_ay, v_dy));
                    __m128 t = _mm_mul_ps(dot, v_invLenSq);
                    t = _mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t));

                    __m128 closestX = _mm_add_ps(v_ax, _mm_mul_ps(t, v_dx));
                    __m128 closestY = _mm_add_ps(v_ay, _mm_mul_ps(t, v_dy));
                    __m128 diffX = _mm_sub_ps(px_v, closestX);
                    __m128 diffY = _mm_sub_ps(py_v_sse, closestY);
                    __m128 distSq = _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY));
                    min_dist_sq = _mm_min_ps(min_dist_sq, distSq);

                    if (edge.yMin <= fy && edge.yMax > fy) {
                        float intersectX = edge.xOfYMin + (fy - edge.yMin) * edge.invSlope;
                        __m128 v_intersect = _mm_set1_ps(intersectX);
                        __m128 cross_mask = _mm_cmpgt_ps(v_intersect, px_v);
                        inside_mask_acc = _mm_xor_ps(inside_mask_acc, cross_mask);
                    }
                }

                __m128 dist_unsigned = _mm_sqrt_ps(min_dist_sq);
                __m128 neg_dist = _mm_sub_ps(ZERO_128, dist_unsigned);
                __m128 sdf = _mm_blendv_ps(dist_unsigned, neg_dist, inside_mask_acc);

                __m128 effectiveFillAlpha = ZERO_128;
                if (drawFill) {
                    __m128 t_fill = _mm_div_ps(_mm_sub_ps(sdf, ANTIALIAS_RANGE_128), ANTIALIAS_RANGE_128);
                    __m128 fillAlpha_raw = _mm_sub_ps(ONE_128, t_fill);
                    effectiveFillAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, fillAlpha_raw)), fillA_sse);
                }

                __m128 effectiveStrokeAlpha = ZERO_128;
                if (drawStroke) {
                    __m128 halfWidthMinusDist = _mm_sub_ps(halfStrokeWidth_sse, dist_unsigned);
                    __m128 t_stroke = _mm_div_ps(halfWidthMinusDist, ANTIALIAS_RANGE_128);
                    effectiveStrokeAlpha = _mm_mul_ps(_mm_max_ps(ZERO_128, _mm_min_ps(ONE_128, t_stroke)), strokeA_sse);
                }

                __m128 finalAlpha, finalR, finalG, finalB;
                if (mode_stroke_over_fill) {
                    __m128 oneMinusStrk = _mm_sub_ps(ONE_128, effectiveStrokeAlpha);
                    __m128 effFillMod = _mm_mul_ps(effectiveFillAlpha, oneMinusStrk);
                    finalAlpha = _mm_add_ps(effectiveStrokeAlpha, effFillMod);

                    finalR = _mm_add_ps(_mm_mul_ps(strokeR_sse, effectiveStrokeAlpha), _mm_mul_ps(fillR_sse, effFillMod));
                    finalG = _mm_add_ps(_mm_mul_ps(strokeG_sse, effectiveStrokeAlpha), _mm_mul_ps(fillG_sse, effFillMod));
                    finalB = _mm_add_ps(_mm_mul_ps(strokeB_sse, effectiveStrokeAlpha), _mm_mul_ps(fillB_sse, effFillMod));

                    __m128 invFinalAlpha = _mm_div_ps(ONE_128, finalAlpha);
                    __m128 zero_mask = _mm_cmpeq_ps(finalAlpha, ZERO_128);
                    invFinalAlpha = _mm_andnot_ps(zero_mask, invFinalAlpha);
                    finalR = _mm_mul_ps(finalR, invFinalAlpha);
                    finalG = _mm_mul_ps(finalG, invFinalAlpha);
                    finalB = _mm_mul_ps(finalB, invFinalAlpha);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    finalR = strokeR_sse; finalG = strokeG_sse; finalB = strokeB_sse;
                }
                else {
                    finalAlpha = effectiveFillAlpha;
                    finalR = fillR_sse; finalG = fillG_sse; finalB = fillB_sse;
                }

                __m128 mask = _mm_cmpgt_ps(finalAlpha, ZERO_128);
                if (_mm_movemask_ps(mask)) {
                    __m128i dest = _mm_loadu_si128((__m128i*) & row[px]);
                    __m128i rgba = blend_pixels_sse(finalAlpha, dest, finalR, finalG, finalB);
                    rgba = _mm_blendv_epi8(dest, rgba, _mm_castps_si128(mask));
                    _mm_storeu_si128((__m128i*) & row[px], rgba);
                }
            }

            // --- Scalar Loop (1-3 pixels) ---
            for (; px <= maxX; ++px) {
                const float fx = static_cast<float>(px) + 0.5f;
                float min_dist_sq = FLT_MAX;
                bool inside = false;

                for (const auto& edge : edges) {
                    // Distance
                    float dx = edge.dx, dy = edge.dy;
                    float t = ((fx - edge.ax) * dx + (fy - edge.ay) * dy) * edge.invLenSq;
                    t = std::max(0.0f, std::min(1.0f, t));
                    float cx = edge.ax + t * dx;
                    float cy = edge.ay + t * dy;
                    float d2 = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy);
                    min_dist_sq = std::min(min_dist_sq, d2);

                    // Inside/Out
                    if (edge.yMin <= fy && edge.yMax > fy) {
                        float intersectX = edge.xOfYMin + (fy - edge.yMin) * edge.invSlope;
                        if (intersectX > fx) inside = !inside;
                    }
                }

                float dist = std::sqrt(min_dist_sq);
                float sdf = inside ? -dist : dist;

                float effectiveFillAlpha = 0.0f;
                if (drawFill) {
                    float t_fill = (sdf - 1.0f) / 1.0f;
                    effectiveFillAlpha = std::max(0.0f, std::min(1.0f, 1.0f - t_fill)) * finalFillOpacity;
                }
                float effectiveStrokeAlpha = 0.0f;
                if (drawStroke) {
                    float t_stroke = (halfStrokeWidth - dist) / 1.0f;
                    effectiveStrokeAlpha = std::max(0.0f, std::min(1.0f, t_stroke)) * finalStrokeOpacity;
                }

                float finalAlpha, R, G, B;
                if (mode_stroke_over_fill) {
                    float strk = effectiveStrokeAlpha;
                    float fill = effectiveFillAlpha * (1.0f - strk);
                    finalAlpha = strk + fill;
                    R = strokeColor.r * (strk / 255.0f) + fillColor.r * (fill / 255.0f);
                    G = strokeColor.g * (strk / 255.0f) + fillColor.g * (fill / 255.0f);
                    B = strokeColor.b * (strk / 255.0f) + fillColor.b * (fill / 255.0f);
                }
                else if (mode_only_stroke) {
                    finalAlpha = effectiveStrokeAlpha;
                    R = strokeColor.r * (finalAlpha / 255.0f);
                    G = strokeColor.g * (finalAlpha / 255.0f);
                    B = strokeColor.b * (finalAlpha / 255.0f);
                }
                else {
                    finalAlpha = effectiveFillAlpha;
                    R = fillColor.r * (finalAlpha / 255.0f);
                    G = fillColor.g * (finalAlpha / 255.0f);
                    B = fillColor.b * (finalAlpha / 255.0f);
                }

                if (finalAlpha > 0.0f) {
                    pa2d::Color src;
                    if (finalAlpha >= 1.0f) {
                        src.r = std::min(255.0f, R * 255.0f); src.g = std::min(255.0f, G * 255.0f); src.b = std::min(255.0f, B * 255.0f); src.a = 255;
                    }
                    else {
                        float inv = 1.0f / finalAlpha;
                        src.r = std::min(255.0f, R * inv * 255.0f); src.g = std::min(255.0f, G * inv * 255.0f); src.b = std::min(255.0f, B * inv * 255.0f); src.a = finalAlpha * 255.0f;
                    }
                    row[px] = Blend(src, row[px]);
                }
            }
        }
    }
}