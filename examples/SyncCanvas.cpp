// PA2D 多窗口协同渲染与物理动画示例
// 版本：1.0.0.beta1 配套示例
//
// 功能说明：
// 1. 演示多窗口协同显示同一动画场景
// 2. 展示基于时间的物理动画（速度、反弹、旋转）
// 3. 演示屏幕捕获与跨窗口渲染
// 4. 展示窗口位置同步和视口裁剪
// 5. 实现高性能的实时动画循环

#include <pa2d.h>
#include <thread>
using namespace pa2d;

// 创建全屏画布作为共享动画背景
Canvas bg(getScreenSize().x, getScreenSize().y);

int main() {
	// 动画矩形对象：包含几何、样式和运动状态
	struct bgRect {
		Rect r;
		Style s;
		Point vel;
		float a_vel;
		bgRect() :s(Style().fill(Color(80, rand() % 256, rand() % 256, rand() % 256))	// 半透明随机颜色，随机圆角
			.radius(5 + rand() % 20)),						 // 随机圆角
			r(rand() % bg.width(), rand() % bg.height()		 // 随机位置,随机尺寸
		, 120 + rand() % 200, 100 + rand() % 250),			 // 随机尺寸
			vel(-26.0f + rand() % 52, -26.0f + rand() % 52)  // 随机速度（ - 25~25）
		, a_vel(5 + rand() % 10){}							 // 随机角速度（5~14度/秒）
	};
	// ==================== 初始化动画对象 ====================
	std::vector<bgRect> rects(50);
	srand(static_cast<unsigned>(time(nullptr)));  // 设置随机种子

	// ==================== 创建多窗口系统 ====================
	// 创建5个窗口(初始位置相同)，初始全部显示
	std::vector<Window> windows(5, Window(400, 300,"").show());

	// ==================== 动画主循环 ====================
	auto last_time = std::chrono::steady_clock::now();
	bool is_closed = false;

	while (!is_closed) {
		// 计算帧时间（秒）
		auto now = std::chrono::steady_clock::now();
		float delta_time = std::chrono::duration<float>(now - last_time).count();
		last_time = now;

		// ==================== 更新动画场景 ====================
		bg.clear(0xFF8AFFF9);  // 清空为浅蓝色背景（ARGB：0xFF8AFFF9）
		for (auto& rect : rects) {
			// 边界碰撞检测与反弹（屏幕边缘）
			// X轴反向
			if ((rect.r.center().x < 0 && rect.vel.x < 0)||
				(rect.r.center().x > bg.width() && rect.vel.x > 0)) rect.vel.x *= -1;
			// Y轴反向
			if ((rect.r.center().y < 0 && rect.vel.y < 0)||
				(rect.r.center().y > bg.height() && rect.vel.y > 0)) rect.vel.y *= -1;

			// 更新位置和旋转，并绘制到背景画布
			// 1. 平移：vel * delta_time（基于时间的位移）
			// 2. 旋转：绕自身中心旋转
			// 3. 绘制：使用预定义的随机样式
			bg.draw(rect.r.translate(rect.vel* delta_time).rotateOnSelf(rect.a_vel* delta_time),rect.s);
		}

		// ==================== 多窗口渲染 ====================
		for (auto& w : windows) {
			// 获取窗口位置和尺寸
			auto pos = w.getPosition();
			auto size =w.getClientSize();

			// 视口裁剪计算与渲染
			w.render(bg,pos.x < 0 ? -pos.x : 0,pos.y < 0 ? -pos.y : 0,
				pos.x < 0 ? 0 : pos.x,pos.y < 0 ? 0 : pos.y,
				pos.x < 0 ? size.width + pos.x : size.width,pos.y < 0 ? size.height + pos.y : size.height);

			// 任一窗口关闭则退出循环
			if (!w.isOpen())is_closed = true;
		}
		// 控制帧率（约60 FPS）
		std::this_thread::sleep_for(std::chrono::milliseconds(16));
	}
	return 0;
}

// 示例说明：
// 1. 时间补偿动画：
//    - 使用delta_time确保动画速度不受帧率影响
//    - 所有运动基于"像素/秒"而非"像素/帧"
//    - 确保在不同性能设备上动画一致
//
// 2. 物理模拟：
//    - 边界碰撞检测（屏幕边缘反弹）
//    - 速度和角速度的独立控制
//    - 基于物理的位置和旋转更新
//
// 3. 多窗口协同：
//    - 多个窗口共享同一动画场景
//    - 每个窗口显示场景的不同视口
//    - 窗口移动时自动更新显示区域
//
// 4. 视口裁剪：
//    - 处理窗口移出屏幕边缘的情况
//    - 动态计算源矩形和目标矩形
//    - 避免渲染不可见区域
//
// 5. 随机化设计：
//    - 颜色、位置、尺寸、速度全部随机
//    - 每个矩形具有独特的运动轨迹
//    - 每次运行产生不同的动画效果
//
// 高级特性展示：
// 1. 高性能渲染：一次计算，多窗口共享
// 2. 时间无关动画：帧率稳定的物理模拟
// 3. 窗口系统集成：位置同步和状态管理
// 4. 内存效率：重用画布避免重复绘制
