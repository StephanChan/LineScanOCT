#ifndef _ZOLIX_ZC300_H_
#define _ZOLIX_ZC300_H_

// 坐标轴
#define AXIS_X									0
#define AXIS_Y									1
#define AXIS_Z									2
// 坐标轴单位
#define UNIT_PP									0
#define UNIT_MM								1
#define UNIT_DEG								2
// 控制器状态
#define STATUS_AXIS_X_LIMIT_P		0x0001
#define STATUS_AXIS_X_LIMIT_N		0x0002
#define STATUS_AXIS_X_ORIGIN		0x0004
#define STATUS_AXIS_Y_LIMIT_P		0x0008
#define STATUS_AXIS_Y_LIMIT_N		0x0010
#define STATUS_AXIS_Y_ORIGIN		0x0020
#define STATUS_AXIS_Z_LIMIT_P		0x0040
#define STATUS_AXIS_Z_LIMIT_N		0x0080
#define STATUS_AXIS_Z_ORIGIN		0x0100
#define STATUS_EMERGENCY			0x0200
#define STATUS_AXIS_X_ALARM		0x0400
#define STATUS_AXIS_Y_ALARM		0x0800
#define STATUS_AXIS_Z_ALARM		0x1000
// 电移台类型
#define STAGE_LINEAR						0
#define STAGE_ROTARY						1
// 归零方式
#define HOME_BY_USER						1
#define HOME_BY_LIMIT						2
#define HOME_BY_ZERO						3
// 移动方式
#define MOVE_ABSOLUTE					0
#define MOVE_RELATIVE					1
#define MOVE_CONTINUOUS			2
// 停止模式
#define STOP_SLOWLY						0
#define STOP_IMMEDIATELY				1
// 蜂鸣器状态
#define BUZZER_SILENT						0
#define BUZZER_BEEP							0

#ifdef __cplusplus
extern "C" {
#endif
#ifdef ZOLIX_ZC_300_EXPORTS
#define ZOLIX_ZC_300_API __declspec(dllexport)
#else
#define ZOLIX_ZC_300_API __declspec(dllimport)
#endif

	ZOLIX_ZC_300_API int zc300_enum_count();
	ZOLIX_ZC_300_API void zc300_enum_info(int index, char* buff, int buff_size);

	ZOLIX_ZC_300_API bool zc300_open(const char * info, const unsigned char addr);
	ZOLIX_ZC_300_API void zc300_close();

	ZOLIX_ZC_300_API bool zc300_get_sn(long * sn);
	ZOLIX_ZC_300_API bool zc300_get_model(char* buff, int buff_size);
	// 读取错误
	ZOLIX_ZC_300_API void zc300_error_info(char* buff, int buff_size);
	// 电移台使能状态(默认使能)
	ZOLIX_ZC_300_API bool zc300_set_enabled(short axis, bool enabled);
	ZOLIX_ZC_300_API bool zc300_get_enabled(short axis, bool * enabled);
	// 电移台类型
	ZOLIX_ZC_300_API bool zc300_set_stage_type(short axis, short type);
	ZOLIX_ZC_300_API bool zc300_get_stage_type(short axis, short * type);
	// IO读写
	ZOLIX_ZC_300_API bool zc300_set_io_output(short output);
	ZOLIX_ZC_300_API bool zc300_get_io_output(short * output);
	ZOLIX_ZC_300_API bool zc300_get_io_input(short * input);
	// 电移台控制参数
	ZOLIX_ZC_300_API bool zc300_set_unit(short axis, short unit);
	ZOLIX_ZC_300_API bool zc300_get_unit(short axis, short * unit);
	ZOLIX_ZC_300_API bool zc300_set_pitch(short axis, float pitch);
	ZOLIX_ZC_300_API bool zc300_get_pitch(short axis, float * pitch);
	ZOLIX_ZC_300_API bool zc300_set_spr(short axis, long spr);
	ZOLIX_ZC_300_API bool zc300_get_spr(short axis, long * spr);
	ZOLIX_ZC_300_API bool zc300_set_ratio(short axis, float ratio);
	ZOLIX_ZC_300_API bool zc300_get_ratio(short axis, float * ratio);
	ZOLIX_ZC_300_API bool zc300_set_init_speed(short axis, float speed);
	ZOLIX_ZC_300_API bool zc300_get_init_speed(short axis, float * speed);
	ZOLIX_ZC_300_API bool zc300_set_move_speed(short axis, float speed);
	ZOLIX_ZC_300_API bool zc300_get_move_speed(short axis, float * speed);
	ZOLIX_ZC_300_API bool zc300_set_acc_speed(short axis, float speed);
	ZOLIX_ZC_300_API bool zc300_get_acc_speed(short axis, float * speed);
	// 电移台归零参数
	ZOLIX_ZC_300_API bool zc300_set_home_speed(short axis, float speed);
	ZOLIX_ZC_300_API bool zc300_get_home_speed(short axis, float * speed);
	ZOLIX_ZC_300_API bool zc300_set_home_mode(short axis, short mode);
	ZOLIX_ZC_300_API bool zc300_get_home_mode(short axis, short * mode);
	// 电移台控制
	ZOLIX_ZC_300_API bool zc300_move(short axis, short type, float distance);
	ZOLIX_ZC_300_API bool zc300_stop(short axis, short mode);
	ZOLIX_ZC_300_API bool zc300_home(short axis);
	// 电移台,控制器状态
	ZOLIX_ZC_300_API bool zc300_set_position(short axis, float pos);
	ZOLIX_ZC_300_API bool zc300_get_position(short axis, float * pos);
	ZOLIX_ZC_300_API bool zc300_get_idle(short axis, bool * idle);
	ZOLIX_ZC_300_API bool zc300_get_status(short * status);
	// 蜂鸣器状态
	ZOLIX_ZC_300_API bool zc300_set_buzzer(short status);
	ZOLIX_ZC_300_API bool zc300_get_buzzer(short * status);
#ifdef __cplusplus
}
#endif
#endif