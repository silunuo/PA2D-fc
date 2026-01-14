#include "../include/geometry/rect.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace pa2d {
    constexpr float GEOMETRY_PI = 3.14159265358979323846f;
    constexpr float GEOMETRY_EPSILON = 1e-6f;

    Rect::Rect() : Shape(GeometryType::RECT), center_(), width_(), height_(), rotation_(),
        cachedVertices_(), verticesDirty_(true) {
    }

    Rect::Rect(float centerX, float centerY, float width, float height, float rotation)
        : Shape(GeometryType::RECT), center_(centerX, centerY), width_(width), height_(height),
        rotation_(rotation), cachedVertices_(), verticesDirty_(true) {
    }

    Rect::Rect(const std::vector<Point>& points) : Shape(GeometryType::RECT) {
        if (points.size() == 4) {
            cachedVertices_[0] = points[0];
            cachedVertices_[1] = points[1];
            cachedVertices_[2] = points[2];
            cachedVertices_[3] = points[3];
            calculateParametersFromVertices();
        }
    }

    Rect& Rect::operator=(const std::vector<Point>& points) {
        if (points.size() == 4) calculateParametersFromVertices();
        return *this;
    }

    Point Rect::center() const { return center_; }
    float Rect::width() const { return width_; }
    float Rect::height() const { return height_; }
    float Rect::rotation() const { return rotation_; }

    Rect& Rect::center(const Point& center) {
        center_ = center;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::center(float x, float y) {
        center_ = Point(x, y);
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::width(float width) {
        width_ = width;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::height(float height) {
        height_ = height;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::rotation(float rotation) {
        rotation_ = rotation;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::translate(float dx, float dy) {
        center_ += Point(dx, dy);
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::translate(Point delta) {
        return translate(delta.x, delta.y);
    }

    Rect& Rect::scale(float factor) {
        width_ *= factor;
        height_ *= factor;
        center_ *= factor;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::scale(float factorX, float factorY) {
        width_ *= factorX;
        height_ *= factorY;
        center_.x *= factorX;
        center_.y *= factorY;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::scaleOnSelf(float factor) {
        width_ *= factor;
        height_ *= factor;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::scaleOnSelf(float factorX, float factorY) {
        width_ *= factorX;
        height_ *= factorY;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::rotate(float angleDegrees) {
        return rotate(angleDegrees, 0, 0);
    }

    Rect& Rect::rotate(float angleDegrees, float centerX, float centerY) {
        float angleRad = angleDegrees * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);

        float dx = center_.x - centerX;
        float dy = center_.y - centerY;

        center_.x = centerX + dx * cosTheta - dy * sinTheta;
        center_.y = centerY + dx * sinTheta + dy * cosTheta;

        rotation_ += angleDegrees;
        verticesDirty_ = true;
        return *this;
    }

    Rect& Rect::rotate(float angleDegrees, Point center) {
        return rotate(angleDegrees, center.x, center.y);
    }

    Rect& Rect::rotateOnSelf(float angleDegrees) {
        rotation_ += angleDegrees;
        verticesDirty_ = true;
        return *this;
    }

    bool Rect::contains(Point point) const {
        float angleRad = -rotation_ * (GEOMETRY_PI / 180.0f);
        float cosTheta = std::cos(angleRad);
        float sinTheta = std::sin(angleRad);
        float dx = point.x - center_.x;
        float dy = point.y - center_.y;
        float localX = dx * cosTheta - dy * sinTheta;
        float localY = dx * sinTheta + dy * cosTheta;
        float halfW = width_ / 2.0f;
        float halfH = height_ / 2.0f;
        return (std::abs(localX) <= halfW) && (std::abs(localY) <= halfH);
    }

    Rect Rect::getBoundingBox() const {
        ensureVerticesUpdated();

        float minX = cachedVertices_[0].x, maxX = cachedVertices_[0].x;
        float minY = cachedVertices_[0].y, maxY = cachedVertices_[0].y;

        for (int i = 1; i < 4; ++i) {
            minX = std::min(minX, cachedVertices_[i].x);
            maxX = std::max(maxX, cachedVertices_[i].x);
            minY = std::min(minY, cachedVertices_[i].y);
            maxY = std::max(maxY, cachedVertices_[i].y);
        }

        return Rect(
            (minX + maxX) * 0.5f,
            (minY + maxY) * 0.5f,
            maxX - minX,
            maxY - minY
        );
    }

    Point Rect::getCenter() const {
        return center_;
    }

    // 迭代器和访问器实现...
    // 由于篇幅限制，这里只展示关键部分

    void Rect::ensureVerticesUpdated() const {
        if (verticesDirty_) {
            const_cast<Rect*>(this)->updateVertices();
            const_cast<Rect*>(this)->verticesDirty_ = false;
        }
    }

    void Rect::updateVertices() const {
        if (!verticesDirty_) return;

        if (width_ <= 0 || height_ <= 0) {
            for (auto& v : cachedVertices_) v = center_;
            verticesDirty_ = false;
            return;
        }

        float halfW = width_ / 2.0f;
        float halfH = height_ / 2.0f;

        float corners[4][2] = {
            {-halfW, -halfH}, { halfW, -halfH},
            { halfW,  halfH}, {-halfW,  halfH}
        };

        if (std::abs(rotation_) > 1e-6f) {
            float angleRad = rotation_ * (GEOMETRY_PI / 180.0f);
            float cosTheta = std::cos(angleRad);
            float sinTheta = std::sin(angleRad);

            for (int i = 0; i < 4; ++i) {
                float x = corners[i][0] * cosTheta - corners[i][1] * sinTheta;
                float y = corners[i][0] * sinTheta + corners[i][1] * cosTheta;
                cachedVertices_[i] = Point(center_.x + x, center_.y + y);
            }
        }
        else {
            cachedVertices_[0] = Point(center_.x - halfW, center_.y - halfH);
            cachedVertices_[1] = Point(center_.x + halfW, center_.y - halfH);
            cachedVertices_[2] = Point(center_.x + halfW, center_.y + halfH);
            cachedVertices_[3] = Point(center_.x - halfW, center_.y + halfH);
        }

        verticesDirty_ = false;
    }

    void Rect::calculateParametersFromVertices() {
        if (cachedVertices_.size() != 4) return;

        // ==================== 1. 计算凸包（确保顶点按逆时针顺序） ====================
        // 复制顶点
        std::array<Point, 4> points = cachedVertices_;

        // 找到最左下角的点作为凸包起点
        auto startIt = std::min_element(points.begin(), points.end(),
            [](const Point& a, const Point& b) {
                return (a.y < b.y) || (a.y == b.y && a.x < b.x);
            });

        // 以最左下角点为参考点进行极角排序
        Point startPoint = *startIt;
        std::sort(points.begin(), points.end(),
            [startPoint](const Point& a, const Point& b) {
                // 计算极角
                float angleA = std::atan2(a.y - startPoint.y, a.x - startPoint.x);
                float angleB = std::atan2(b.y - startPoint.y, b.x - startPoint.x);

                // 如果极角相同，按距离排序
                if (std::abs(angleA - angleB) < 1e-6f) {
                    float distA = (a.x - startPoint.x) * (a.x - startPoint.x) +
                        (a.y - startPoint.y) * (a.y - startPoint.y);
                    float distB = (b.x - startPoint.x) * (b.x - startPoint.x) +
                        (b.y - startPoint.y) * (b.y - startPoint.y);
                    return distA < distB;
                }
                return angleA < angleB;
            });

        // Graham扫描构建凸包
        std::array<Point, 4> hull;
        hull[0] = points[0];
        hull[1] = points[1];
        int hullSize = 2;

        auto cross = [](const Point& a, const Point& b, const Point& c) {
            return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
            };

        for (int i = 2; i < 4; ++i) {
            while (hullSize >= 2 &&
                cross(hull[hullSize - 2], hull[hullSize - 1], points[i]) <= 0) {
                hullSize--;
            }
            hull[hullSize++] = points[i];
        }

        // 如果是退化情况（共线点），直接计算
        if (hullSize < 3) {
            // 退化为线段
            float minX = points[0].x, maxX = points[0].x;
            float minY = points[0].y, maxY = points[0].y;
            for (const auto& p : points) {
                minX = std::min(minX, p.x);
                maxX = std::max(maxX, p.x);
                minY = std::min(minY, p.y);
                maxY = std::max(maxY, p.y);
            }
            center_.x = (minX + maxX) * 0.5f;
            center_.y = (minY + maxY) * 0.5f;
            width_ = maxX - minX;
            height_ = maxY - minY;
            rotation_ = 0.0f;
            verticesDirty_ = true;
            return;
        }

        // ==================== 2. 旋转卡壳算法求最小面积矩形 ====================
        // 初始化变量
        float minArea = std::numeric_limits<float>::max();
        float bestWidth = 0, bestHeight = 0;
        float bestAngle = 0;
        Point bestCenter;

        // 对于凸包的每条边作为矩形的一边
        for (int i = 0; i < hullSize; ++i) {
            int j = (i + 1) % hullSize;

            // 当前边的向量
            Point edge = hull[j] - hull[i];
            float edgeLength = std::hypot(edge.x, edge.y);
            if (edgeLength < 1e-6f) continue;

            // 计算当前边的角度
            float angle = std::atan2(edge.y, edge.x);

            // 旋转坐标系，使当前边水平
            float cosA = std::cos(-angle);
            float sinA = std::sin(-angle);

            // 计算所有点在旋转后的坐标系中的坐标
            std::array<float, 4> rotatedX, rotatedY;
            float minX = std::numeric_limits<float>::max();
            float maxX = -std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxY = -std::numeric_limits<float>::max();

            for (int k = 0; k < hullSize; ++k) {
                // 平移，使hull[i]为原点
                float dx = hull[k].x - hull[i].x;
                float dy = hull[k].y - hull[i].y;

                // 旋转
                rotatedX[k] = dx * cosA - dy * sinA;
                rotatedY[k] = dx * sinA + dy * cosA;

                // 更新边界
                minX = std::min(minX, rotatedX[k]);
                maxX = std::max(maxX, rotatedX[k]);
                minY = std::min(minY, rotatedY[k]);
                maxY = std::max(maxY, rotatedY[k]);
            }

            // 当前旋转下的矩形参数
            float width = maxX - minX;
            float height = maxY - minY;
            float area = width * height;

            // 计算中心点在旋转后的坐标系中
            float centerX = (minX + maxX) * 0.5f;
            float centerY = (minY + maxY) * 0.5f;

            // 旋转回原坐标系
            float cosA_inv = std::cos(angle);
            float sinA_inv = std::sin(angle);
            Point center;
            center.x = centerX * cosA_inv - centerY * sinA_inv + hull[i].x;
            center.y = centerX * sinA_inv + centerY * cosA_inv + hull[i].y;

            // 如果找到更小的面积，更新结果
            if (area < minArea) {
                minArea = area;
                bestWidth = width;
                bestHeight = height;
                bestAngle = angle * (180.0f / GEOMETRY_PI);
                bestCenter = center;
            }
        }

        // ==================== 3. 检查是否为有效矩形 ====================
        if (minArea < 1e-6f || bestWidth < 1e-6f || bestHeight < 1e-6f) {
            // 退化情况：点共线或重合
            float minX = points[0].x, maxX = points[0].x;
            float minY = points[0].y, maxY = points[0].y;
            for (const auto& p : points) {
                minX = std::min(minX, p.x);
                maxX = std::max(maxX, p.x);
                minY = std::min(minY, p.y);
                maxY = std::max(maxY, p.y);
            }
            center_.x = (minX + maxX) * 0.5f;
            center_.y = (minY + maxY) * 0.5f;
            width_ = maxX - minX;
            height_ = maxY - minY;
            rotation_ = 0.0f;
        }
        else {
            // 正常情况
            center_ = bestCenter;
            width_ = bestWidth;
            height_ = bestHeight;
            rotation_ = bestAngle;

            // 确保角度在[0, 180)范围内
            while (rotation_ < 0) rotation_ += 180.0f;
            while (rotation_ >= 180.0f) rotation_ -= 180.0f;

            // 确保宽度 >= 高度，角度对应宽度方向
            if (width_ < height_) {
                std::swap(width_, height_);
                rotation_ += 90.0f;
                while (rotation_ >= 180.0f) rotation_ -= 180.0f;
            }
        }

        verticesDirty_ = true;
    }
}