use scap::{capturer::Capturer, frame::Frame};

use std::sync::mpsc;
use std::thread::sleep;
use std::time::Duration;

use opencv::prelude::*;
use opencv::{
    core::{self, Scalar},
    highgui, imgcodecs, imgproc,
};

mod handwriting;
use handwriting::{HandWriting, Writer};

mod parser;

#[allow(dead_code)]
mod consts {
    use opencv::core::Scalar;

    pub const BLACK: Scalar = Scalar::new(0.0, 0.0, 0.0, 255.0);
    pub const WHITE: Scalar = Scalar::new(255.0, 255.0, 255.0, 255.0);
    pub const GRAY: Scalar = Scalar::new(128.0, 128.0, 128.0, 255.0);
    pub const LIGHT_GRAY: Scalar = Scalar::new(192.0, 192.0, 192.0, 255.0);
    pub const DARK_GRAY: Scalar = Scalar::new(64.0, 64.0, 64.0, 255.0);
}

#[derive(Debug, Clone)]
struct Template {
    pub name: String,
    mat: Mat,
}

fn loading_templates(black_and_white: bool) -> Vec<Template> {
    println!("Loading templates from ./assets/templates");
    // read pngs from dir
    let path = std::path::Path::new("./assets/templates");
    let mut res = vec![];
    for entry in std::fs::read_dir(path).unwrap() {
        if let Ok(entry) = entry {
            if entry.file_type().is_ok_and(|t| t.is_file()) {
                let file_name: String = entry.file_name().into_string().unwrap();
                if file_name.ends_with(".png") {
                    println!("{}", file_name);
                    let name = file_name[..file_name.len() - 4].to_string();
                    if let Ok(img) = opencv::imgcodecs::imread(
                        entry.path().to_str().unwrap(),
                        imgcodecs::IMREAD_UNCHANGED,
                    ) {
                        if !black_and_white {
                            res.push(Template { name, mat: img });
                        } else {
                            let mut mat = Mat::default();
                            if let Err(e) =
                                core::in_range(&img, &consts::BLACK, &consts::LIGHT_GRAY, &mut mat)
                            {
                                println!("error converting to gray: {}", e);
                                continue;
                            }
                            res.push(Template { name, mat });
                        }
                    } else {
                        println!("error loading template png: {}", name);
                    }
                } else {
                    println!("not a png file: {}", file_name);
                }
            } else {
                println!("not a file: {:?}", entry);
            }
        }
    }
    res
}

fn init_scap() {
    // Check if the platform is supported
    if !scap::is_supported() {
        panic!("❌ Platform not supported");
    }
    // Check if we have permission to capture screen
    // If we don't, request it.
    if !scap::has_permission() {
        println!("❌ Permission not granted. Requesting permission...");
        if !scap::request_permission() {
            panic!("❌ Permission denied");
        } else {
            println!("✅ Permission granted");
        }
    }
}

fn update_template(templates: &mut Vec<Mat>, roi: &Mat) -> bool {
    let mut sims = vec![];
    for template in templates.iter() {
        // 计算两个 Mat 之间的 L2 范数 (也可以使用其他范数)
        let mut diff = Mat::default();
        core::absdiff(roi, template, &mut diff).unwrap();
        let dif = core::norm(&diff, core::NORM_L2, &Mat::default()).unwrap();
        let similarity = dif / (diff.total() as f64); // 归一化相似度
        if similarity < 1.2 {
            // 如果相似度低于阈值，认为两者相似，不需要更新
            return false;
        }
        sims.push(similarity);
    }
    println!("Similarity {:?}", sims);
    // 如果没有找到相似的模板，则将新 roi 添加到模板集
    templates.push(roi.clone());
    true
}

// return index to the most similar templates
// or -1 if all is greater than 2.0
fn find_template(templates: &Vec<Template>, roi: &Mat) -> (i32, f64) {
    let mut res = (-1, 2.0);
    for (i, template) in templates.iter().enumerate() {
        // 计算两个 Mat 之间的 L2 范数 (也可以使用其他范数)
        let mut diff = Mat::default();
        // println!("Sizes: {:?} & {:?}", template.mat.size(), roi.size());
        // println!(
        //     "channels: {:?} & {:?}",
        //     template.mat.channels(),
        //     roi.channels()
        // );
        core::absdiff(roi, &template.mat, &mut diff).unwrap();
        let dif = core::norm(&diff, core::NORM_L2, &Mat::default()).unwrap();
        let similarity = dif / (diff.total() as f64); // 归一化相似度
        if similarity < res.1 {
            // 如果相似度低阈值，认为两者相似，更新
            res.0 = i as i32;
            res.1 = similarity;
        }
    }
    res
}

fn extract_template(config: Config) {
    // highgui::named_window("window", highgui::WINDOW_GUI_EXPANDED).unwrap();
    highgui::named_window("window", highgui::WINDOW_GUI_NORMAL).unwrap();
    // highgui::imshow("window", &t[0].mat).unwrap();

    let mut capturer = Capturer::new(scap::capturer::Options {
        fps: 2,
        output_type: scap::frame::FrameType::BGRAFrame,
        show_cursor: false,
        source_rect: Some(scap::capturer::Area {
            origin: scap::capturer::Point {
                x: config.pos.0 as f64,
                y: config.pos.1 as f64,
            },
            size: scap::capturer::Size {
                width: config.size.0 as f64,
                height: config.size.1 as f64,
            },
        }),
        ..Default::default()
    });

    let writing = HandWriting::default();
    let mut writer = Writer::default();

    let mut number_index = 0;
    let mut templates = Vec::<Mat>::new();

    capturer.start_capture();
    loop {
        let key = highgui::wait_key(1).unwrap();
        // sleep(Duration::from_millis(500));
        let Ok(Frame::BGRA(mut b)) = capturer.get_next_frame() else {
            panic!("Error")
        };
        let mut img = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                b.height,
                b.width,
                core::CV_8UC4,
                b.data.as_mut_ptr() as *mut std::ffi::c_void,
                core::Mat_AUTO_STEP,
            )
        }
        .expect("can not create Mat!");
        // 裁剪出这个子矩形区域
        let width = config.size.0 as i32;
        let height = (config.size.1 * 0.14) as i32;
        let roi = core::Rect::new(0, (config.size.1 * 0.18) as i32, width, height);
        let mut cropped = Mat::roi_mut(&mut img, roi).unwrap();
        // 二值化
        let mut mask = Mat::default();
        core::in_range(&cropped, &consts::BLACK, &consts::DARK_GRAY, &mut mask).unwrap();
        // 查找并切割数字
        let mut column_projection = vec![0; config.size.0 as usize];
        // 遍历图像的每一列，统计白色像素的数量
        for y in 0..height {
            for x in 0..width {
                let pixel_value = *mask.at_2d::<u8>(y, x).unwrap(); // 获取像素值
                if pixel_value == 255 {
                    column_projection[x as usize] += 1;
                }
            }
        }
        // 寻找连续区域，裁剪
        let mut start = 0;
        for i in 2..(width - 1) as usize {
            if start == 0 {
                if column_projection[i] > 0 {
                    start = i - 2;
                }
            } else if column_projection[i] == 0 {
                // println!("Colum from {} to {} ", start, i + 1);
                // end = i + 1
                let roi = core::Rect::new(start as i32, 0, (i + 2 - start) as i32, height);
                start = 0;
                // Space
                if key == 32 {
                    // 裁剪出数字区域
                    let number_roi = Mat::roi(&cropped, roi).unwrap();
                    // 缩放新的 ROI 到标准尺寸
                    let mut resized_roi = Mat::default();
                    imgproc::resize(
                        &number_roi,
                        &mut resized_roi,
                        core::Size_ {
                            width: 40,
                            height: 40,
                        },
                        0.0,
                        0.0,
                        opencv::imgproc::INTER_LINEAR,
                    )
                    .unwrap();
                    if update_template(&mut templates, &resized_roi) {
                        println!("Detect new template!");
                        // 保存每个数字为单独的文件
                        let filename = format!("./assets/temp/number_{number_index}.png");
                        number_index += 1;
                        imgcodecs::imwrite(&filename, &resized_roi, &core::Vector::<i32>::new())
                            .unwrap();
                    }
                } else {
                    imgproc::rectangle(
                        &mut cropped,
                        roi.clone(),
                        [255.0, 0.0, 0.0, 0.0].into(),
                        2,
                        imgproc::LINE_8,
                        0,
                    );
                    highgui::imshow("window", &cropped).unwrap();
                }
            }
        }

        if key == 113 {
            // quit with q
            break;
        }
        // 0
        if key >= '0' as i32 && key <= '9' as i32 {
            let key = key as u8 as char;
            writer.write(&writing, &config, key.to_string());
        }
    }
}

fn compare_template(templates: Vec<Template>, config: Config, sender: mpsc::Sender<String>) {
    let mut capturer = Capturer::new(scap::capturer::Options {
        fps: 2,
        output_type: scap::frame::FrameType::BGRAFrame,
        show_cursor: false,
        source_rect: Some(scap::capturer::Area {
            origin: scap::capturer::Point {
                x: config.pos.0 as f64,
                y: config.pos.1 as f64,
            },
            size: scap::capturer::Size {
                width: config.size.0 as f64,
                height: config.size.1 as f64,
            },
        }),
        ..Default::default()
    });
    let mut last_signal = String::new();

    capturer.start_capture();
    loop {
        let key = highgui::wait_key(1).unwrap();
        if key == 113 || key == 27 {
            drop(sender);
            // quit with q or esc
            return;
        }
        // sleep(Duration::from_millis(500));
        let Ok(Frame::BGRA(mut b)) = capturer.get_next_frame() else {
            panic!("Error")
        };
        let mut img = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                b.height,
                b.width,
                core::CV_8UC4,
                b.data.as_mut_ptr() as *mut std::ffi::c_void,
                core::Mat_AUTO_STEP,
            )
        }
        .expect("can not create Mat!");
        // 裁剪出这个子矩形区域
        let width = config.size.0 as i32;
        let height = (config.size.1 * 0.14) as i32;
        let roi = core::Rect::new(0, (config.size.1 * 0.18) as i32, width, height);
        let mut cropped = Mat::roi_mut(&mut img, roi).unwrap();
        // 二值化
        let mut mask = Mat::default();
        core::in_range(&cropped, &consts::BLACK, &consts::DARK_GRAY, &mut mask).unwrap();
        // 查找并切割数字
        let mut column_projection = vec![0; config.size.0 as usize];
        // 遍历图像的每一列，统计白色像素的数量
        for y in 0..height {
            for x in 0..width {
                let pixel_value = *mask.at_2d::<u8>(y, x).unwrap(); // 获取像素值
                if pixel_value == 255 {
                    column_projection[x as usize] += 1;
                }
            }
        }
        // 寻找连续区域，裁剪
        let mut start = 0;
        let mut signal = String::new();
        for i in 2..(width - 1) as usize {
            if start == 0 {
                if column_projection[i] > 0 {
                    start = i - 2;
                }
            } else if column_projection[i] == 0 {
                // println!("Colum from {} to {} ", start, i + 1);
                // end = i + 1
                let roi = core::Rect::new(start as i32, 0, (i + 2 - start) as i32, height);
                start = 0;
                // 裁剪出数字区域
                let number_roi = Mat::roi(&cropped, roi).unwrap();
                // 缩放新的 ROI 到标准尺寸
                let mut resized_roi = Mat::default();
                imgproc::resize(
                    &number_roi,
                    &mut resized_roi,
                    core::Size_ {
                        width: 40,
                        height: 40,
                    },
                    0.0,
                    0.0,
                    opencv::imgproc::INTER_LINEAR,
                )
                .unwrap();
                // highgui::imshow("window", &resized_roi);
                // println!("channels: {:?}", resized_roi.channels());
                // continue;
                let (index, _) = find_template(&templates, &resized_roi);
                if index == -1 {
                    println!("Cannot match!");
                    continue;
                } else {
                    match templates[index as usize].name.as_str() {
                        "plus" => signal += "+",
                        "minus" => signal += "-",
                        "times" => signal += "*",
                        "divide" => signal += "/",
                        "percent" => signal += "%",
                        "left" => signal += "(",
                        "right" => signal += ")",
                        "equal" => signal += "=",
                        "x" => signal += "x",
                        "comma" => signal += ",",
                        "dot" => signal += ".",
                        "etc" => signal += "`",
                        "e4" => signal += "W",
                        "e8l" => signal += "Y",
                        "e8r" => signal += "i",
                        c if u8::from_str_radix(c, 10).is_ok_and(|d| 0 <= d && d <= 9) => {
                            signal += c
                        }
                        c => {
                            println!("Unknown symbol name: {}", c);
                            println!("It may be added in next version!");
                            continue;
                        }
                    }
                }
            }
        }

        if !signal.is_empty() && last_signal != signal {
            last_signal = signal.clone();
            sender.send(signal).unwrap();
        }
    }
}

fn main() {
    init_scap();
    let templates = loading_templates(false);
    println!("Templates Lodaed");
    // println!("Templates {:?}", t);

    let config = Config::detect_phone_position();
    println!("Find app screen: {:#?}", config);

    // extract_template(config);

    let cfg = config.clone();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || compare_template(templates, cfg, sender));

    let writing = HandWriting::default();
    let mut writer = Writer::default();

    let mut capturer = Capturer::new(scap::capturer::Options {
        fps: 2,
        output_type: scap::frame::FrameType::BGRAFrame,
        show_cursor: false,
        source_rect: Some(scap::capturer::Area {
            origin: scap::capturer::Point {
                x: config.handwriting_pos.0 as f64,
                y: config.handwriting_pos.1 as f64,
            },
            size: scap::capturer::Size {
                width: config.handwriting_size.0 as f64,
                height: config.handwriting_size.1 as f64,
            },
        }),
        ..Default::default()
    });
    if let Ok(Frame::BGRA(mut b)) = {
        capturer.start_capture();
        let c = capturer.get_next_frame();
        capturer.stop_capture();
        c
    } {
        let mat = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                b.height,
                b.width,
                core::CV_8UC4,
                b.data.as_mut_ptr() as *mut std::ffi::c_void,
                core::Mat_AUTO_STEP,
            )
        }
        .expect("can not create Mat!");
        imgcodecs::imwrite("1.png", &mat, &core::Vector::<i32>::new()).unwrap();
        println!("Showing");
    }
    while let Ok(expression) = receiver.recv() {
        println!("Recognize expression: {}", expression);
        let res = parser::parse(expression);
        if !res.is_empty() {
            println!("Result: {}", res);
            writer.write(&writing, &config, res);
        }
    }
    println!("Ternimate!")
}

#[derive(Debug, Clone)]
struct Config {
    /// the starting pos of phone in screen
    pub pos: (f32, f32),
    /// the size of phone in screen
    pub size: (f32, f32),
    /// current problem's center height releative to screenshot
    pub current: f32,
    /// next problem's center height releative to screenshot
    pub upcoming: f32,
    /// handwriting region starting pos in screen
    pub handwriting_pos: (f32, f32),
    /// handwriting region size in screen
    pub handwriting_size: (f32, f32),
    pub display_scale: f32,
    pub font_size: f32,

    /// speed
    pub line_speed: i32,
    pub arc_speed: i32,
}

impl Config {
    fn detect_phone_position() -> Self {
        let mut capturer = Capturer::new(scap::capturer::Options {
            output_type: scap::frame::FrameType::BGRAFrame,
            ..Default::default()
        });
        let [pr, dr] = loop {
            if let Ok(Frame::BGRA(mut b)) = {
                capturer.start_capture();
                let c = capturer.get_next_frame();
                capturer.stop_capture();
                c
            } {
                let mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        b.height,
                        b.width,
                        core::CV_8UC4,
                        b.data.as_mut_ptr() as *mut std::ffi::c_void,
                        core::Mat_AUTO_STEP,
                    )
                }
                .expect("can not create Mat!");
                let mut img_hsv = Mat::default();
                imgproc::cvt_color(&mat, &mut img_hsv, imgproc::COLOR_BGR2HSV, 0).unwrap();
                let lower_bound = core::Scalar::new(20.0, 178.0, 253.0, 0.0); // 橙色的下界 (HSV)
                let upper_bound = core::Scalar::new(22.0, 222.0, 255.0, 0.0); // 橙色的上界 (HSV)
                let mut mask = Mat::default();
                core::in_range(&img_hsv, &lower_bound, &upper_bound, &mut mask).unwrap();
                let [pr, dr] = Self::find_rectangle(&mask);
                println!("problem {:?}, drawing {:?}", pr, dr);
                // square size not correct
                if pr.2 < 200 || pr.3 < 70 || dr.2 < 200 || dr.3 < 70 {
                    println!("Cannot capture app screen, try again!");
                    sleep(Duration::from_secs(2));
                    continue;
                }
                if (pr.0 - dr.0).abs() > 100 || pr.1 + pr.3 * 3 / 2 > dr.1 || pr.1 + pr.3 * 4 < dr.1
                {
                    println!("Cannot capture app screen at correct position, try again!");
                    sleep(Duration::from_secs(2));
                    continue;
                }
                // highgui::imshow("window", &mat).unwrap();
                break [pr, dr];
            } else {
                println!("Error cannot capture")
            }
        };
        Self {
            pos: (pr.0 as f32, pr.1 as f32),
            size: (pr.2 as f32, pr.3 as f32 * 2.0),
            current: pr.3 as f32 * 0.5,
            upcoming: pr.3 as f32 * 1.5,
            handwriting_pos: (dr.0 as f32, dr.1 as f32),
            handwriting_size: (dr.2 as f32, dr.3 as f32),
            display_scale: 1.25,
            font_size: 120.0,
            line_speed: 8,
            arc_speed: 4,
        }
    }

    fn find_rectangle(mask: &Mat) -> [(i32, i32, i32, i32); 2] {
        // 定义形态学内核 (kernel) 的大小，决定了对细线和小噪声的清除程度
        let kernel = Mat::ones(8, 8, core::CV_8U).unwrap().to_mat().unwrap();
        let mut cleared_mask = Mat::default();
        // 第一步：应用开运算 (Opening)，去除噪声和细小的物体
        imgproc::morphology_ex(
            &mask,
            //  img,
            &mut cleared_mask,
            imgproc::MORPH_OPEN,
            &kernel,
            core::Point::new(-1, -1),
            1, // 迭代次数
            core::BORDER_CONSTANT,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        // 查找轮廓
        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(
            // img,
            &cleared_mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )
        .unwrap();

        // 遍历轮廓，找到最大矩形的轮廓
        let mut problem_area = (0, 0, 0, 0);
        for contour in contours.iter() {
            // imgproc::polylines(img, &contour, true, green, 2, imgproc::LINE_8, 0).unwrap();
            let mut approx = core::Vector::<core::Point>::new();
            imgproc::approx_poly_dp(
                &contour,
                &mut approx,
                0.02 * imgproc::arc_length(&contour, true).unwrap(),
                true,
            )
            .unwrap();
            // 如果是四边形（矩形）
            if approx.len() == 4 {
                let rect = imgproc::bounding_rect(&approx).unwrap();
                if rect.area() > problem_area.2 * problem_area.3 {
                    problem_area.0 = rect.x;
                    problem_area.1 = rect.y;
                    problem_area.2 = rect.width;
                    problem_area.3 = rect.height;
                }
                println!("Found rectangle at: {:?}", rect);
            } else {
                println!("Found contour with {} points", approx.len());
            }
        }

        // 第二步：应用闭运算 (Closing)，连接虚线框
        let kernel = Mat::ones(16, 16, core::CV_8U).unwrap().to_mat().unwrap();
        imgproc::morphology_ex(
            &mask,
            //  img,
            &mut cleared_mask,
            imgproc::MORPH_DILATE,
            &kernel,
            core::Point::new(-1, -1),
            1, // 迭代次数
            core::BORDER_CONSTANT,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();
        // 查找轮廓
        let mut contours: core::Vector<core::Vector<core::Point_<i32>>> =
            core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(
            // img,
            &cleared_mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            core::Point::new(0, 0),
        )
        .unwrap();

        // 遍历轮廓，找到最大矩形的轮廓
        let mut drawing_area = (
            problem_area.0,
            problem_area.1 + 2 * problem_area.3,
            problem_area.2,
            problem_area.3,
        );
        for contour in contours.iter() {
            let mut approx = core::Vector::<core::Point>::new();
            imgproc::approx_poly_dp(
                &contour,
                &mut approx,
                0.02 * imgproc::arc_length(&contour, true).unwrap(),
                true,
            )
            .unwrap();
            // 如果是四边形（矩形）
            if approx.len() == 4 {
                let rect = imgproc::bounding_rect(&approx).unwrap();
                if rect.area() > drawing_area.2 * drawing_area.3 {
                    drawing_area.0 = rect.x;
                    drawing_area.1 = rect.y;
                    drawing_area.2 = rect.width;
                    drawing_area.3 = rect.height;
                }
                println!("Found rectangle at: {:?}", rect);
            } else {
                println!("Found contour with {} points", approx.len());
            }
        }
        [problem_area, drawing_area]
    }
}
