
#include "voxel_mapping.hpp"
#include "meshing/mesh_rec_display.hpp"
#include "meshing/mesh_rec_geometry.hpp"
#include "tools/tools_thread_pool.hpp"
#include <shared_mutex>
#include "meshing/r3live/pointcloud_rgbd.hpp"
#include <fstream>

extern Global_map       g_map_rgb_pts_mesh;
extern Triangle_manager g_triangles_manager;
extern int              g_current_frame;

extern double                       minimum_pts;
extern double                       g_meshing_voxel_size;
extern FILE *                       g_fp_cost_time;
extern FILE *                       g_fp_lio_state;
extern bool                         g_flag_pause;
extern const int                    number_of_frame;
extern int                          appending_pts_frame;
extern LiDAR_frame_pts_and_pose_vec g_eigen_vec_vec;

// 定义两个原子类型 —— 防止 append加点操作 比 mesh重建部分快很多
std::atomic<long> append_id{0};
std::atomic<long> mesh_id{0};


int        g_maximum_thread_for_rec_mesh;
std::mutex g_mutex_append_map;
std::mutex g_mutex_reconstruct_mesh;
std::mutex g_mutex_pts_vector;

extern double g_LiDAR_frame_start_time;
double        g_vx_map_frame_cost_time;
static double g_LiDAR_frame_avg_time;


/////////////////////// 新增部分 ////////////////

extern std::shared_mutex g_mutex_eigen_vec_vec;     // 为保护g_eigen_vec_vec的读写操作

std::mutex g_mutex_colored_points;
std::mutex g_mutex_projection_points;
std::shared_ptr<std::shared_future<void> > g_render_thread = nullptr;
extern LiDAR_color_frame_pts_and_pose_vec g_eigen_color_vec_vec;
namespace {
    std::unique_ptr<std::thread> g_pub_thr = nullptr;
    int flag = 0;
}

///////////////////////////////////

struct Rec_mesh_data_package
{
    pcl::PointCloud< pcl::PointXYZI >::Ptr m_frame_pts;
    Eigen::Quaterniond                     m_pose_q;
    Eigen::Vector3d                        m_pose_t;
    int                                    m_frame_idx;
    Rec_mesh_data_package( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, Eigen::Quaterniond pose_q, Eigen::Vector3d pose_t, int frame_idx )
    {
        m_frame_pts = frame_pts;
        m_pose_q = pose_q;
        m_pose_t = pose_t;
        m_frame_idx = frame_idx;
    }
};

std::mutex                                  g_mutex_data_package_lock;
std::list< Rec_mesh_data_package >          g_rec_mesh_data_package_list;
std::shared_ptr< Common_tools::ThreadPool > g_thread_pool_rec_mesh = nullptr;

extern int                                  g_enable_mesh_rec;
extern int                                  g_save_to_offline_bin;

LiDAR_frame_pts_and_pose_vec       g_ponintcloud_pose_vec;

namespace{
    const double image_obs_cov = 1.5;
    const double process_noise_sigma = 0.15;
}
std::ofstream file("frame_ids.txt", std::ios::app);
// 函数重载 - 使用所有数据的mesh重建函数 - 包含点云渲染以及mesh重建
void incremental_mesh_reconstruction( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, cv::Mat img, Eigen::Quaterniond pose_q, Eigen::Vector3d pose_t, int frame_idx )
{
    // g_flag_pause应该就是一个与GUI界面联动部分
    while ( g_flag_pause )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    if(frame_pts == nullptr)
    {
        LOG(ERROR) << "incremental_mesh_reconstruction : frame_pts is nullptr";
        return;
    }

    if (file.is_open()) {
        file << "frame_idx: " << frame_idx <<" | incremental_mesh_reconstruction | frame_pts size: " << frame_pts->points.size() << std::endl;
    }

    /*** 打包点云数据 - 这部分用于GUI中的点云信息的生成 ***/
    Eigen::Matrix< double, 7, 1 > pose_vec;
    pose_vec.head< 4 >() = pose_q.coeffs().transpose();
    pose_vec.block( 4, 0, 3, 1 ) = pose_t;

    std::unique_lock lock(g_mutex_eigen_vec_vec);
    for ( int i = 0; i < frame_pts->points.size(); i++ )
    {
        g_eigen_vec_vec[ frame_idx ].first.emplace_back( frame_pts->points[ i ].x, frame_pts->points[ i ].y, frame_pts->points[ i ].z,
                                                         frame_pts->points[ i ].intensity );
    }
    g_eigen_vec_vec[ frame_idx ].second = pose_vec;
    lock.unlock();


    int append_point_step = std::max( ( int ) 1, ( int ) std::round( frame_pts->points.size() / appending_pts_frame ) );
    Common_tools::Timer tim, tim_total;
    // g_map_rgb_pts_mesh 对应的为全局地图 | 上锁可能是由于 incremental_mesh_reconstruction 这个函数是在线程池中进行commit_task执行的 | 数据送入全局地图之后并没有进行降采样
        // g_mutex_append_map 这个互斥锁是保护g_map_rgb_pts_mesh类本身的操作 —— 防止其他线程来执行这部分的操作

    /// @bug 有时候也会出现这里有问题的debug信息 | debug中频繁出现的Finish the append_points_to_global_map提示是在这里直接return了 但是frame_id却没有更新(应该是进入了这个函数但是我估计没有成功修改掉)
    g_mutex_append_map.lock();
    if( !(g_map_rgb_pts_mesh.append_points_to_global_map( *frame_pts, frame_idx, nullptr, append_point_step )))
    {
//        ++append_id;
        g_mutex_append_map.unlock();
        LOG(ERROR) << "[frame_idx]:" << frame_idx << " || There is a memory fault in append_points_to_global_map ";
        return;
    }
    ++append_id;
    g_mutex_append_map.unlock();
    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the append_points_to_global_map ";

    g_map_rgb_pts_mesh.m_mutex_m_box_recent_hitted->lock();
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited = g_map_rgb_pts_mesh.m_voxels_recent_visited;
    g_map_rgb_pts_mesh.m_mutex_m_box_recent_hitted->unlock();

    /*** 位姿变换 ***/
    Eigen::Matrix3d rot_i2w = pose_q.toRotationMatrix();
    Eigen::Vector3d pos_i2w = pose_t;

    /*** m_extR: l2i || m_camera_ext_R: c2l ***/
    Eigen::Matrix3d R_w2c;
    R_w2c = rot_i2w * extR * camera_ext_R;
    Eigen::Vector3d t_w2c;
    t_w2c = rot_i2w * extR * camera_ext_t + extR * extT + pos_i2w;

    std::shared_ptr<Image_frame> image_pose = std::make_shared<Image_frame>(cam_k);
    image_pose->set_pose(eigen_q(R_w2c), t_w2c);
    image_pose->m_img = img;
//        LOG(INFO) << "image_pose->m_img.rows" << image_pose->m_img.rows;
    image_pose->m_timestamp = ros::Time::now().toSec();
    image_pose->init_cubic_interpolation();
    image_pose->image_equalize();

    /// @attention 突然感觉这个函数的作用不大 - 基本上是不需要被使用的(这个生成的数据后面也没有被使用)
//    std::vector<cv::Point2f> pts_2d_vec;
//    std::vector<std::shared_ptr<RGB_pts> > rgb_pts_vec;
////    g_mutex_projection_points.lock();
//    g_map_rgb_pts_mesh.selection_points_for_projection(image_pose, &rgb_pts_vec, &pts_2d_vec, 10);
//    g_mutex_projection_points.unlock();
//    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the selection_points_for_projection ";
    /// TODO 尝试一下将rendering的全过程写在这个函数里面而不是使用函数调用 —— 或许是因为函数调用的时候出现了问题(主要是看mesh重建是直接放在这个函数里面了)

    /*** 渲染部分 ***/
//    g_render_thread = std::make_shared< std::shared_future< void > >( g_thread_pool_rec_mesh->commit_task(
//            render_pts_in_voxels_mp, image_pose, &g_map_rgb_pts_mesh.m_voxels_recent_visited, image_pose->m_timestamp ) );

    //    auto idx = frame_idx;
    /// @bug render_pts_in_voxels_mp的voxel_ptr有问题(越界导致段错误) - 所以我怀疑是输入的voxel_ptr集合(即voxels_recent_visited)出现问题 (貌似不是这个部分的问题...没找到)
    // 因为这里会操作全局地图 中的部分voxel的m_ptr_in_grid这个函数, 所以上锁


//    render_pts_in_voxels_mp(image_pose, &g_map_rgb_pts_mesh.m_voxels_recent_visited , image_pose->m_timestamp );
//    render_pts_in_voxels_mp(image_pose, &voxels_recent_visited , image_pose->m_timestamp );

    long numbers_of_voxels = voxels_recent_visited.size();
    std::vector<shared_ptr<RGB_Voxel>> voxels_for_render;
    for ( Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end(); it++ )
    {
        voxels_for_render.push_back( *it );
    }

    image_pose->m_acc_render_count = 0;
    image_pose->m_acc_photometric_error = 0;

//    cv::parallel_for_( cv::Range( 0, numbers_of_voxels ),
//                       [&]( const cv::Range& r ) { thread_render_pts_in_voxel( r.start, r.end, image_pose, &voxels_for_render, image_pose->m_timestamp ); } );
//    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the rendering ";


    // 模仿immesh里面mesh重建的部分 —— 直接在这里写成tbb的加速 | 因为这个部分只会调用全局地图中的点云数据,所以上的是与之前相同的锁
//    std::atomic<long> my_render_pts_count;

    try
    {
        /// @attention 这里使用值传递与引用传递的区别有多少 —— 多线程里面两者是不是有些区别
        cv::parallel_for_(cv::Range(0, numbers_of_voxels), [=](const cv::Range &r) {
            vec_3 pt_w;
            vec_3 rgb_color;
            double u, v;
            double pt_cam_norm;
            g_mutex_append_map.lock();
            for (int voxel_idx = r.start; voxel_idx < r.end; voxel_idx++) {
                RGB_voxel_ptr voxel_ptr = voxels_for_render[voxel_idx];
                for (int pt_idx = 0; pt_idx < voxel_ptr->m_pts_in_grid.size(); pt_idx++) {
                    pt_w = voxel_ptr->m_pts_in_grid[pt_idx]->get_pos();
                    if (image_pose->project_3d_point_in_this_img(pt_w, u, v, nullptr, 1.0) == false)
                        continue;

                    pt_cam_norm = (pt_w - image_pose->m_pose_w2c_t).norm();
                    // 在图像上获取点云的颜色信息 | 然后对这个voxel中的所有点云的颜色信息进行更新
                    rgb_color = image_pose->get_rgb(u, v, 0);
                    if (voxel_ptr->m_pts_in_grid[pt_idx]->update_rgb(
                            rgb_color, pt_cam_norm, vec_3(image_obs_cov, image_obs_cov, image_obs_cov),
                            image_pose->m_timestamp)) {
//                        my_render_pts_count++;
                    }
                }
            }
            g_mutex_append_map.unlock();
        });
        LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the rendering ";
    }
    catch ( ... )
    {
        for ( int i = 0; i < 100; i++ )
        {
            cout << ANSI_COLOR_RED_BOLD << "Exception in tbb parallels...in rendering" << ANSI_COLOR_RESET << endl;
        }
        return;
    }


    /*** 发布线程 ***/
//    g_map_rgb_pts_mesh.m_last_updated_frame_idx++;
//    if( g_pub_thr == nullptr  && flag == 0 )
//    {
//        // 对于10000个RGB点构建一个彩色点云的发布器，然后创建出多个发布器之后再发布数据(不过不理解的部分在与这里明明是进行数据的读取，为什么这里不需要上锁-也访问全局地图了)
//        g_pub_thr = std::make_unique<std::thread>(&Global_map::service_pub_rgb_maps, &g_map_rgb_pts_mesh);
//        flag = 1;           // 因为创建线程写在了while()循环中, 为了避免线程的重复创建，这里设置flag
//    }

    /*** 完成render之后的mesh重建过程 ***/
//    std::atomic< int >    voxel_idx( 0 );
    std::mutex mtx_triangle_lock, mtx_single_thr;

//     为什么 迭代器 放在了tbb外面(反正没有使用 —— 基本没有问题 )
    typedef std::unordered_set< std::shared_ptr< RGB_Voxel > >::iterator set_voxel_it;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     removed_triangle_list;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     added_triangle_list;

    g_mutex_reconstruct_mesh.lock();
    tim.tic();
    tim_total.tic();

    try
    {
        /// @attention Cpp的并行计算库tbb(又相当于是创建了新的多个线程来处理数据) | parallel_for_each 需要指定一个搜索范围+一个lambda表达式 | lambda的形参 对应就是从voxels_recent_visited取出来的voxel数据
        tbb::parallel_for_each( voxels_recent_visited.begin(), voxels_recent_visited.end(), [ =, &removed_triangle_list, &added_triangle_list, &mtx_triangle_lock ]( const std::shared_ptr< RGB_Voxel > &voxel ) {
            if ( ( voxel->m_meshing_times >= 1 ) || ( voxel->m_new_added_pts_count < 0 ) )
            {
                return;
            }
            Common_tools::Timer tim_lock;
            tim_lock.tic();
            voxel->m_meshing_times++;
            voxel->m_new_added_pts_count = 0;   // 点进入全局地图的时候就会被增加

            vec_3 pos_1 = vec_3( voxel->m_pos[ 0 ], voxel->m_pos[ 1 ], voxel->m_pos[ 2 ] );     // 这里获取到的数据是voxel的位置 但是一致没有被使用
            // printf("Voxels [%d], (%d, %d, %d) ", count, pos_1(0), pos_1(1), pos_1(2) );

            // 每一次执行时定义的临时变量 | 获取neighbor_voxels中的所有特征点(现在还没有去获取周围voxel的特征点) | 如果仅仅获取当前voxel中的RGB点,是不是不需要上锁(因为一次只访问单独的一个voxel，不同线程之间不会有冲突)
            std::unordered_set< std::shared_ptr< RGB_Voxel > > neighbor_voxels;
            neighbor_voxels.insert( voxel );


            g_mutex_append_map.lock();
            std::vector< RGB_pt_ptr > pts_in_voxels = retrieve_pts_in_voxels( neighbor_voxels );
            if ( pts_in_voxels.size() < 3 )
            {
                g_mutex_append_map.unlock();
                return;
            }
            g_mutex_append_map.unlock();

            // Voxel-wise mesh pull
//            g_mutex_pts_vector.lock();
            pts_in_voxels = retrieve_neighbor_pts_kdtree( pts_in_voxels );
            // 这里虽然是删除, 但是不会影响原始数据(所以与rendering线程崩溃的部分没有关系)
            pts_in_voxels = remove_outlier_pts( pts_in_voxels, voxel );

            // pts_in_voxels 对应的是所有RGB点 | relative_point_indices 获取到所有点的id号
            std::set< long > relative_point_indices;
            for ( RGB_pt_ptr tri_ptr : pts_in_voxels )
            {
                relative_point_indices.insert( tri_ptr->m_pt_index );
            }

            int iter_count = 0;
            g_triangles_manager.m_enable_map_edge_triangle = 0;

            // 更新数据 (不知道为什么要进行这种操作) | 相当于重新生成了点云数据 | m_rgb_pts_vec 对应Global_map中所有点的管理器
            pts_in_voxels.clear();
            for ( auto p : relative_point_indices )
            {
                pts_in_voxels.push_back( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] );
            }


            std::set< long > convex_hull_index, inner_pts_index;
//          //   mtx_triangle_lock.lock();

            /*** 三角剖分 ***/
            /*
             * 1. 注意这里说的是都是一些2D结构: 凸包点在是外围的点 内部点是在凸包点构成几何形状中的点
             * 2. convex_hull_index + inner_pts_index 保存的对应的点的id号 | 调用CGAL算法进行三角剖分
             * 3. add_triangle_idx作为返回值 | 关于上面的convex_hull_index + inner_pts_index只是划分一个2D平面上点的分布, 生成mesh的点可能是convex也可能是inner
             *    add_triangle_idx 为当前 voxel 中的所有点都进行了三角剖分之后的结果
             * */
            voxel->m_short_axis.setZero();
            std::vector< long > add_triangle_idx = delaunay_triangulation( pts_in_voxels, voxel->m_long_axis, voxel->m_mid_axis,
                                                                           voxel->m_short_axis, convex_hull_index, inner_pts_index );
//            // 当前 pts_in_voxels 已经是包含了在其他voxel中的点 | 这里在全局地图中设置点属于哪一个voxel (而没有在RGB_Voxel中放入其他voxel中的点)
            for ( auto p : inner_pts_index )
            {
                if ( voxel->if_pts_belong_to_this_voxel( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] ) )
                {
                    g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = true;
                    g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
                }
            }
            // 是不是convex_hull_index中的点的作用不 这里给一个false之后就不用管了
            for ( auto p : convex_hull_index )
            {
                g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = false;
                g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
            }

//            // 原代码中上锁上的总感觉位置不是很正确, 这里多移动一部分, 一直移动到 包含所有使用g_rgb_pt_mesh中的点云地图的部分
//            g_mutex_append_map.unlock();
//            g_mutex_pts_vector.unlock();

            /** 从现在开始, 整个系统中最重要的数据结构出现了 —— Triangle_manager 在后续中被使用 **/
            /// TODO 这里需要整理一些 triangle的顶点信息 - 主要这些顶点的颜色信息是在哪里生成的
            // triangles_sets 为这些点目前对应的所有triangle信息
            Triangle_set triangles_sets = g_triangles_manager.find_relative_triangulation_combination( relative_point_indices );
            Triangle_set triangles_to_remove, triangles_to_add, existing_triangle;
            // Voxel-wise mesh commit(所有应该构建出来的Mesh与已经存在的mesh对比)
            triangle_compare( triangles_sets, add_triangle_idx, triangles_to_remove, triangles_to_add, &existing_triangle );
//            triangle_compare( triangles_sets, add_triangle_idx, triangles_to_remove, triangles_to_add );        // triangles_to_add 保留了当前所有的triangle数据
            // Refine normal index - 将新增的triangle以及已经存在的triangle进行修正 | 修正是整个平面的法向量
            for ( auto triangle_ptr : triangles_to_add )
            {
                // 使用g_eigen_vec_vec的位姿信息 - 即对应了在GUI做展示的时候 实际运行出来的实际 camera 的位姿信息 | 短轴部分对应的应该是2D投影平面上的法向量
                std::shared_lock lock(g_mutex_eigen_vec_vec);
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }

            for ( auto triangle_ptr : existing_triangle )
            {
                std::shared_lock lock(g_mutex_eigen_vec_vec);
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }
            std::unique_lock< std::mutex > lock( mtx_triangle_lock );   // 对removed_triangle_list以及added_triangle_list进行上锁 保证tbb执行的时候不会出现同时读写问题
            // 这里只是做了关于add与remove的整理 将一个voxel与其对应的triangle进行了整理 | 具体要处理的部分还是要在pull模块处理掉
            removed_triangle_list.emplace( std::make_pair( voxel, triangles_to_remove ) );
            added_triangle_list.emplace( std::make_pair( voxel, triangles_to_add ) );
//            existing_triangle_list.emplace( std::make_pair( voxel, existing_triangle ) );

//            voxel_idx++;
        } );

        LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the delaunay_triangulation + triangle_compare + correct_triangle_index";

    }
    catch ( ... )
    {
        for ( int i = 0; i < 100; i++ )
        {
            cout << ANSI_COLOR_RED_BOLD << "Exception in tbb parallels...in mesh reconstruction" << ANSI_COLOR_RESET << endl;
        }
        return;
    }

    /*** tbb mesh重建部分已经完成 - 剩余部分就是对整体进行处理 ***/
    double              mul_thr_cost_time = tim.toc( " ", 0 );
    Common_tools::Timer tim_triangle_cost;
    int                 total_delete_triangle = 0, total_add_triangle = 0;
    // Voxel-wise mesh push
    for ( auto &triangles_set : removed_triangle_list )
    {
        // triangles_set中保存的是 (体素:体素中的triangle集合) 所以这里取出来的是一个体素中的triangle集合
        total_delete_triangle += triangles_set.second.size();
        g_triangles_manager.remove_triangle_list( triangles_set.second );
    }
//    LOG(INFO) << "[incremental_mesh_reconstruction]: The count of deleted triangle is "<< total_delete_triangle;

    /// @attention 每一个voxel都不保存triangle数据 - 直接使用全局的triangle_manager来管理
        // added_triangle_list 中包含着所有 voxel 需要被处理的数据
    for ( auto &triangle_list : added_triangle_list )
    {
        Triangle_set triangle_idx = triangle_list.second;
        total_add_triangle += triangle_idx.size();
        // 相当于是单独处理每一个voxel中的triangle数据
        for ( auto triangle_ptr : triangle_idx )
        {
            double tmp_color[3][3] = {0};
            // 补充生成三角顶点颜色 | j代表是第几个顶点
//            for(auto j = 0; j < 3; ++j)
//            {
//                tmp_color[j][0] = g_map_rgb_pts_mesh.m_rgb_pts_vec[ triangle_ptr->m_tri_pts_id[ j ] ]->m_rgb[0];
//                tmp_color[j][1] = g_map_rgb_pts_mesh.m_rgb_pts_vec[ triangle_ptr->m_tri_pts_id[ j ] ]->m_rgb[1];
//                tmp_color[j][2] = g_map_rgb_pts_mesh.m_rgb_pts_vec[ triangle_ptr->m_tri_pts_id[ j ] ]->m_rgb[2];
//            }
            // 输入Triangle三个顶点的数据 | 主要是有很重要的Synchronize_triangle的数据转换
//            Triangle_ptr tri_ptr = g_triangles_manager.insert_triangle( triangle_ptr->m_tri_pts_id[ 0 ], triangle_ptr->m_tri_pts_id[ 1 ],
//                                                                        triangle_ptr->m_tri_pts_id[ 2 ], tmp_color, 1 );
            Triangle_ptr tri_ptr = g_triangles_manager.insert_triangle( triangle_ptr->m_tri_pts_id[ 0 ], triangle_ptr->m_tri_pts_id[ 1 ],
                                                                        triangle_ptr->m_tri_pts_id[ 2 ], 1 );
            tri_ptr->m_index_flip = triangle_ptr->m_index_flip;
        }
    }
    ++mesh_id;
    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish the insert_triangle";
    g_mutex_reconstruct_mesh.unlock();

//    if ( g_fp_cost_time )
//    {
//        if ( frame_idx > 0 )
//            g_LiDAR_frame_avg_time = g_LiDAR_frame_avg_time * ( frame_idx - 1 ) / frame_idx + ( g_vx_map_frame_cost_time ) / frame_idx;
//        fprintf( g_fp_cost_time, "%d %lf %d %lf %lf\r\n", frame_idx, tim.toc( " ", 0 ), ( int ) voxel_idx.load(), g_vx_map_frame_cost_time,
//                 g_LiDAR_frame_avg_time );
//        fflush( g_fp_cost_time );
//    }
    if ( g_current_frame < frame_idx )
    {
        g_current_frame = frame_idx;
    }
    else
    {
        if ( g_eigen_vec_vec[ g_current_frame + 1 ].second.size() > 7 )
        {
            g_current_frame++;
        }
    }
}



/// @brief 输入进来的数据就是一帧点云的全部信息 | 以及这里对应的投影部分并没有使用voxelmap中的数据而是从这里自己进行计算
void incremental_mesh_reconstruction( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, Eigen::Quaterniond pose_q, Eigen::Vector3d pose_t, int frame_idx )
{
    // g_flag_pause应该就是一个与GUI界面联动部分
    while ( g_flag_pause )
    {
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    }

    Eigen::Matrix< double, 7, 1 > pose_vec;
    pose_vec.head< 4 >() = pose_q.coeffs().transpose();
    pose_vec.block( 4, 0, 3, 1 ) = pose_t;

//    std::unique_lock lock(g_mutex_eigen_vec_vec);
    // g_eigen_vec_vec把每一个点都转换成为矩阵 () | 将一帧的所有点云 与 当前lidar的位置设置了映射关系
    for ( int i = 0; i < frame_pts->points.size(); i++ )
    {
        g_eigen_vec_vec[ frame_idx ].first.emplace_back( frame_pts->points[ i ].x, frame_pts->points[ i ].y, frame_pts->points[ i ].z,
                                                         frame_pts->points[ i ].intensity );
    }
    g_eigen_vec_vec[ frame_idx ].second = pose_vec;
//    lock.unlock();

    // appending_pts_frame : 一帧frame中有多少点可以被加入到global map中 | append_point_step 使用自适应的步长(限制每一帧点云中应该有多少点可以被加入到地图中)
    int                 append_point_step = std::max( ( int ) 1, ( int ) std::round( frame_pts->points.size() / appending_pts_frame ) );
    Common_tools::Timer tim, tim_total;


    // g_map_rgb_pts_mesh 对应的为全局地图
    g_mutex_append_map.lock();
    g_map_rgb_pts_mesh.append_points_to_global_map( *frame_pts, frame_idx, nullptr, append_point_step );
    // 知道哪些voxel需要被处理(这个数据会跟临时定义的voxels_recent_visited变量进行数据互换，所以没有显示地让整个数据清空的操作)
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited = g_map_rgb_pts_mesh.m_voxels_recent_visited;
    g_mutex_append_map.unlock();
    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish append_points_to_global_map";

    /// @attention 原子类型 - 相当与是mutex锁的平替(因为锁在开关的时候比较耗费时间，一般都是针对一段代码进行Mutex操作，而atomic是对单一变量的操作 - 多个线程都能访问这个数据)
    std::atomic< int >    voxel_idx( 0 );
    std::mutex mtx_triangle_lock, mtx_single_thr;
    typedef std::unordered_set< std::shared_ptr< RGB_Voxel > >::iterator set_voxel_it;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     removed_triangle_list;
    std::unordered_map< std::shared_ptr< RGB_Voxel >, Triangle_set >     added_triangle_list;

    g_mutex_reconstruct_mesh.lock();
    tim.tic();
    tim_total.tic();
    try
    {
        /// @attention Cpp的并行计算库tbb(又相当于是创建了新的多个线程来处理数据) | parallel_for_each 需要指定一个搜索范围+一个lambda表达式
            // 将这些所有voxel数据分配给这些线程中进行处理(其会自动进行分区计算) | 形参直接使用lambda表达式 [ & ]表示直接这个函数可以引用的方式直接访问外部变量 | voxel作为形参出现，读取到的是voxels_recent_visited中的数据
            // tbb自动地进行分配 voxels_recent_visited 空间进行并行计算
        tbb::parallel_for_each( voxels_recent_visited.begin(), voxels_recent_visited.end(), [ & ]( const std::shared_ptr< RGB_Voxel > &voxel ) {
            // std::unique_lock<std::mutex> thr_lock(mtx_single_thr);
            // printf_line;
            if ( ( voxel->m_meshing_times >= 1 ) || ( voxel->m_new_added_pts_count < 0 ) )
            {
                return;
            }

            Common_tools::Timer tim_lock;
            tim_lock.tic();
            voxel->m_meshing_times++;
            voxel->m_new_added_pts_count = 0;   // 点进入全局地图的时候就会被增加

            vec_3 pos_1 = vec_3( voxel->m_pos[ 0 ], voxel->m_pos[ 1 ], voxel->m_pos[ 2 ] );

            // printf("Voxels [%d], (%d, %d, %d) ", count, pos_1(0), pos_1(1), pos_1(2) );
            std::unordered_set< std::shared_ptr< RGB_Voxel > > neighbor_voxels;
            neighbor_voxels.insert( voxel );

            g_mutex_append_map.lock();
            // 获取neighbor_voxels中的所有特征点(现在还没有去获取周围的特征点) | 如果仅仅获取当前voxel中的RGB点,是不是不需要上锁(因为一次只访问单独的一个voxel，不同线程之间不会有冲突)
            std::vector< RGB_pt_ptr > pts_in_voxels = retrieve_pts_in_voxels( neighbor_voxels );
            if ( pts_in_voxels.size() < 3 )
            {
                g_mutex_append_map.unlock();
                return;
            }
            g_mutex_append_map.unlock();


            // Voxel-wise mesh pull
                // pts_in_voxels 一开始获取到这个数据的时候是需要上锁的 - 但是现在其变成了一个lambda函数中的临时变量,就不需要上锁了
            pts_in_voxels = retrieve_neighbor_pts_kdtree( pts_in_voxels );
            pts_in_voxels = remove_outlier_pts( pts_in_voxels, voxel );

            std::set< long > relative_point_indices;
            for ( RGB_pt_ptr tri_ptr : pts_in_voxels )
            {
                relative_point_indices.insert( tri_ptr->m_pt_index );
            }

            int iter_count = 0;
            g_triangles_manager.m_enable_map_edge_triangle = 0;

            // 更新数据 (不知道为什么要进行这种操作)
            pts_in_voxels.clear();
            for ( auto p : relative_point_indices )
            {
                // 只读取也不要上锁 g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]
                pts_in_voxels.push_back( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] );
            }
           
            std::set< long > convex_hull_index, inner_pts_index;

            voxel->m_short_axis.setZero();
            std::vector< long > add_triangle_idx = delaunay_triangulation( pts_in_voxels, voxel->m_long_axis, voxel->m_mid_axis,
                                                                               voxel->m_short_axis, convex_hull_index, inner_pts_index );

            // 之前pts_in_voxels已经是包含了物体位置在其他voxel中的点，这里在全局地图中设置点属于哪一个voxel(而没有在RGB_Voxel中放入其他voxel中的点)
            for ( auto p : inner_pts_index )
            {
                if ( voxel->if_pts_belong_to_this_voxel( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] ) )
                {
                    g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = true;
                    g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
                }
            }

            for ( auto p : convex_hull_index )
            {
                g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_is_inner_pt = false;
                g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ]->m_parent_voxel = voxel;
            }

            /** 从现在开始, 整个系统中最重要的数据结构出现了 —— Triangle_manager 在后续中被使用 **/
            Triangle_set triangles_sets = g_triangles_manager.find_relative_triangulation_combination( relative_point_indices );
            Triangle_set triangles_to_remove, triangles_to_add, existing_triangle;
            // Voxel-wise mesh commit(所有应该构建出来的Mesh与已经存在的mesh对比)
            triangle_compare( triangles_sets, add_triangle_idx, triangles_to_remove, triangles_to_add, &existing_triangle );
            
            // Refine normal index(这里对应的为实际需要被增加进来的数据)
            for ( auto triangle_ptr : triangles_to_add )
            {
//                std::shared_lock lock(g_mutex_eigen_vec_vec);
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }
            for ( auto triangle_ptr : existing_triangle )
            {
//                std::shared_lock lock(g_mutex_eigen_vec_vec);
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }

            std::unique_lock< std::mutex > lock( mtx_triangle_lock );
            // 这里只是做了什么需要add 什么需要remove的整理 | 具体要处理的部分还是要在pull模块处理掉
            removed_triangle_list.emplace( std::make_pair( voxel, triangles_to_remove ) );
            added_triangle_list.emplace( std::make_pair( voxel, triangles_to_add ) );
            
            voxel_idx++;
        } );
    }
    catch ( ... )
    {
        for ( int i = 0; i < 100; i++ )
        {
            cout << ANSI_COLOR_RED_BOLD << "Exception in tbb parallels..." << ANSI_COLOR_RESET << endl;
        }
        return;
    }

    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish mesh reconstruction";
    double              mul_thr_cost_time = tim.toc( " ", 0 );
    Common_tools::Timer tim_triangle_cost;
    int                 total_delete_triangle = 0, total_add_triangle = 0;
    // Voxel-wise mesh push
    for ( auto &triangles_set : removed_triangle_list )
    {
        // triangles_set中保存的是 (体素:体素中的triangle集合) 所以这里取出来的是一个体素中的triangle集合
        total_delete_triangle += triangles_set.second.size();
        g_triangles_manager.remove_triangle_list( triangles_set.second );
    }
//    LOG(INFO) << "[incremental_mesh_reconstruction]: The count of deleted triangle is "<< total_delete_triangle;

    // 这里面的每个元素对应着一个voxel中的triangle集合
    for ( auto &triangle_list : added_triangle_list )
    {
        Triangle_set triangle_idx = triangle_list.second;
        total_add_triangle += triangle_idx.size();
        // 对于这个voxel中triangle的每一个元素进行处理
        for ( auto triangle_ptr : triangle_idx )
        {
            // 这个函数对应的数据比较重要，mesh的可视化部分主要使用这里添加的数据
            Triangle_ptr tri_ptr = g_triangles_manager.insert_triangle( triangle_ptr->m_tri_pts_id[ 0 ], triangle_ptr->m_tri_pts_id[ 1 ],
                                                                        triangle_ptr->m_tri_pts_id[ 2 ], 1 );
            tri_ptr->m_index_flip = triangle_ptr->m_index_flip;
        }
    }
    g_mutex_reconstruct_mesh.unlock();

    LOG(INFO) << "[frame_idx]:" << frame_idx <<" Finish triangle insert";
    if ( g_fp_cost_time )
    {
        if ( frame_idx > 0 )
            g_LiDAR_frame_avg_time = g_LiDAR_frame_avg_time * ( frame_idx - 1 ) / frame_idx + ( g_vx_map_frame_cost_time ) / frame_idx;
        fprintf( g_fp_cost_time, "%d %lf %d %lf %lf\r\n", frame_idx, tim.toc( " ", 0 ), ( int ) voxel_idx.load(), g_vx_map_frame_cost_time,
                 g_LiDAR_frame_avg_time );
        fflush( g_fp_cost_time );
    }
    if ( g_current_frame < frame_idx )
    {
        g_current_frame = frame_idx;
    }
    else
    {
        if ( g_eigen_vec_vec[ g_current_frame + 1 ].second.size() > 7 )
        {
            g_current_frame++;
        }
    }
}



/// @brief 新建/使用线程池,不断处理数据 | 整理处理的数据更换成为 g_rec_color_data_package_list 包含点云 图像以及位姿数据的部分 - 这个线程里面让点的渲染与mesh重建一起使用 | 因为mesh里面打包数据的时候就可以打包颜色信息 - 所以这里让渲染之后再处理mesh重建

void service_reconstruct_mesh()
{
    LOG(INFO) << "Starting the mesh thread";
//    std::ofstream file("frame_ids.txt", std::ios::app);
    // 创建一个线程池
    if ( g_thread_pool_rec_mesh == nullptr )
    {
        // 不需要手动删除(线程池自己会进行销毁工作) —— 在线程池类的析构函数中进行数据的销毁
        LOG(INFO) << "The total thread of Thread pool is: " << g_maximum_thread_for_rec_mesh;
        g_thread_pool_rec_mesh = std::make_shared< Common_tools::ThreadPool >( g_maximum_thread_for_rec_mesh );
    }
    int drop_frame_num = 0;
    while ( 1 ) {
        // 等待数据
//            while ( g_rec_mesh_data_package_list.size() == 0 )
//            {
//                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
//            }
        while (g_rec_color_data_package_list.size() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 这里上锁是因为service_lidar_update在另一个线程中会放入数据，而在这个线程要取出数据所以上锁
//            g_mutex_data_package_lock.lock();
        // 维持数据量
//            while ( g_rec_mesh_data_package_list.size() > 1e5 )
//            {
//                cout << "Drop mesh frame [" << g_rec_mesh_data_package_list.front().m_frame_idx;
//                printf( "], total_drop = %d, all_frame = %d\r\n", drop_frame_num++, g_rec_mesh_data_package_list.front().m_frame_idx );
//                g_rec_mesh_data_package_list.pop_front();
//            }

        // 目前需要处理的数据量
//            if ( g_rec_mesh_data_package_list.size() > 10 )
//            {
//                LOG(INFO) << "Poor real-time performance, current buffer size = " <<  g_rec_mesh_data_package_list.size();
////                cout << "Poor real-time performance, current buffer size = " << g_rec_mesh_data_package_list.size() << endl;
//            }

        g_mutex_all_data_package_lock.lock();
        Point_clouds_color_data_package data_pack_front;

        if (g_rec_color_data_package_list.size() > 30)
        {
            LOG(INFO) << "Poor real-time performance, current buffer size = " << g_rec_color_data_package_list.size();
        }


        // 手动控制 —— 玄学认为 全局加点 与 mesh重建线程的区别是线程池出现问题的根源
        if ( append_id - mesh_id >= 0 && append_id - mesh_id <= 12)
        {
            data_pack_front = g_rec_color_data_package_list.front();
            g_rec_color_data_package_list.pop_front();
            // 最重要的数据不能为空
            if(data_pack_front.m_frame_pts == nullptr || data_pack_front.m_img.empty() || data_pack_front.m_frame_pts->points.size() == 0)
            {
                g_mutex_all_data_package_lock.unlock();
                continue;
            }
            else
                g_mutex_all_data_package_lock.unlock();
        }
        else
        {
            std::this_thread::sleep_for( std::chrono::microseconds( 50 ) );
            g_mutex_all_data_package_lock.unlock();
            continue;
        }


//            Rec_mesh_data_package data_pack_front = g_rec_mesh_data_package_list.front();
//            g_rec_mesh_data_package_list.pop_front();
//            g_mutex_data_package_lock.unlock();

            // ANCHOR - Comment follow line to disable meshing
            if ( g_enable_mesh_rec )
            {
                // 线程池开始运行 | 给定的参数分别为 : 任务函数 + 一些参数
//                g_thread_pool_rec_mesh->commit_task( incremental_mesh_reconstruction, data_pack_front.m_frame_pts, data_pack_front.m_pose_q,
//                                                     data_pack_front.m_pose_t, data_pack_front.m_frame_idx );

//                g_thread_pool_rec_mesh->commit_task([&]() {
//                    incremental_mesh_reconstruction(data_pack_front.m_frame_pts,
//                                                    data_pack_front.m_pose_q,
//                                                    data_pack_front.m_pose_t,
//                                                    data_pack_front.m_frame_idx);
//                });

                /// @bug 全靠gpt修改出来的代码 神奇 本来我都想去不再使用这个线程池,直接处理数据了
                    // 由于这部分使用线程池,在这里又是一个while循环 - 可能出现incremental_mesh_reconstruction同时被多处执行
                    g_thread_pool_rec_mesh->commit_task([&]() {
                    incremental_mesh_reconstruction(data_pack_front.m_frame_pts,
                                                    data_pack_front.m_img,
                                                    data_pack_front.m_pose_q,
                                                    data_pack_front.m_pose_t,
                                                    data_pack_front.m_frame_idx);

                    if (file.is_open()) {
                        file << "m_frame_idx: " << data_pack_front.m_frame_idx << " | append_id: " << append_id << " | mesh_id: " << mesh_id << std::endl;
                    } else {
                        std::cerr << "Unable to open file for writing frame IDs" << std::endl;
                    }
                });

//                // 这里感觉是在线程池里面重复执行这个函数的时候导致的错误
//                incremental_mesh_reconstruction(data_pack_front.m_frame_pts,
//                                                data_pack_front.m_img,
//                                                data_pack_front.m_pose_q,
//                                                data_pack_front.m_pose_t,
//                                                data_pack_front.m_frame_idx);

            }
        //
        std::this_thread::sleep_for( std::chrono::microseconds( 50 ) );
    }
    file.close();
}

extern bool  g_flag_pause;
int          g_frame_idx = 0;
std::thread *g_rec_mesh_thr = nullptr;

void start_mesh_threads( int maximum_threads = 20 )
{
    // g_eigen_vec_vec将每一帧的所有点以及其对应的位姿数据全部保存起来
    if ( g_eigen_vec_vec.size() <= 0 )
    {
        g_eigen_vec_vec.resize( 1e6 );
    }

    if ( g_rec_mesh_thr == nullptr )
    {
        g_maximum_thread_for_rec_mesh = maximum_threads;    // 只是一个int数据，对应其数据量的值
        // 启动mesh重建线程线程 - 启动函数(但是这里并没有调用join函数结束这个线程)
        g_rec_mesh_thr = new std::thread( service_reconstruct_mesh );
    }
}

// 这个函数貌似只有在直接提供点云数据的时候才会被使用
void reconstruct_mesh_from_pointcloud( pcl::PointCloud< pcl::PointXYZI >::Ptr frame_pts, double minimum_pts_distance )
{
    start_mesh_threads();
    cout << "=== reconstruct_mesh_from_pointcloud ===" << endl;
    cout << "Input pointcloud have " << frame_pts->points.size() << " points." << endl;
    pcl::PointCloud< pcl::PointXYZI >::Ptr all_cloud_ds( new pcl::PointCloud< pcl::PointXYZI > );

    pcl::VoxelGrid< pcl::PointXYZI > sor;
    sor.setInputCloud( frame_pts );
    sor.setLeafSize( minimum_pts_distance, minimum_pts_distance, minimum_pts_distance );
    sor.filter( *all_cloud_ds );

    cout << ANSI_COLOR_BLUE_BOLD << "Raw points number = " << frame_pts->points.size()
         << ", downsampled points number = " << all_cloud_ds->points.size() << ANSI_COLOR_RESET << endl;
    g_mutex_data_package_lock.lock();
    g_rec_mesh_data_package_list.emplace_back( all_cloud_ds, Eigen::Quaterniond::Identity(), vec_3::Zero(), 0 );
    g_mutex_data_package_lock.unlock();
}

void open_log_file()
{
    if ( g_fp_cost_time == nullptr || g_fp_lio_state == nullptr )
    {
        Common_tools::create_dir( std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output" ).c_str() );
        std::string cost_time_log_name = std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output/mesh_cost_time.log" );
        std::string lio_state_log_name = std::string( Common_tools::get_home_folder() ).append( "/ImMesh_output/lio_state.log" );
        // cout << ANSI_COLOR_BLUE_BOLD ;
        // cout << "Record cost time to log file:" << cost_time_log_name << endl;
        // cout << "Record LIO state to log file:" << cost_time_log_name << endl;
        // cout << ANSI_COLOR_RESET;
        g_fp_cost_time = fopen( cost_time_log_name.c_str(), "w+" );
        g_fp_lio_state = fopen( lio_state_log_name.c_str(), "w+" );
    }
}

std::vector< vec_4 > convert_pcl_pointcloud_to_vec( pcl::PointCloud< pcl::PointXYZI > &pointcloud )
{
    int                  pt_size = pointcloud.points.size();
    std::vector< vec_4 > eigen_pt_vec( pt_size );
    for ( int i = 0; i < pt_size; i++ )
    {
        eigen_pt_vec[ i ]( 0 ) = pointcloud.points[ i ].x;
        eigen_pt_vec[ i ]( 1 ) = pointcloud.points[ i ].y;
        eigen_pt_vec[ i ]( 2 ) = pointcloud.points[ i ].z;
        eigen_pt_vec[ i ]( 3 ) = pointcloud.points[ i ].intensity;
    }
    return eigen_pt_vec;
}


/// @brief
/*
 * 1. voxelmap 的重建(包含位姿估计)
 * 2. 关于mesh图的重建部分 在这里只有启动新线程进行处理(至少12个)
 * 3. 最后保留了一些点云数据
 * PS: 这里直接读取之前模块已经计算好的R,t信息
 *
 * */

void Voxel_mapping::map_incremental_grow()
{
    start_mesh_threads( m_meshing_maximum_thread_for_rec_mesh );
    if ( m_use_new_map )
    {
        while ( g_flag_pause )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }
        // startTime = clock();

        // world_lidar为降采样之后的数据 | world_lidar_full 为完整的一帧点云数据
        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar_full( new pcl::PointCloud< pcl::PointXYZI > );

        std::vector< Point_with_var > pv_list;
        // TODO: saving pointcloud to file
        // pcl::io::savePCDFileBinary(Common_tools::get_home_folder().append("/r3live_output/").append("last_frame.pcd").c_str(), *m_feats_down_body);

        transformLidar( state.rot_end, state.pos_end, m_feats_down_body, world_lidar );

        for ( size_t i = 0; i < world_lidar->size(); i++ )
        {
            Point_with_var pv;
            pv.m_point << world_lidar->points[ i ].x, world_lidar->points[ i ].y, world_lidar->points[ i ].z;

            /// @bug 输出协方差出现了问题 | 用原始公式计算的协方差一直是存在一些问题的
            M3D point_crossmat = m_cross_mat_list[ i ];
            M3D var = m_body_cov_list[ i ];
            var = ( state.rot_end * m_extR ) * var * ( state.rot_end * m_extR ).transpose() +
                  ( -point_crossmat ) * state.cov.block< 3, 3 >( 0, 0 ) * ( -point_crossmat ).transpose() + state.cov.block< 3, 3 >( 3, 3 );
            pv.m_var = var;
            pv_list.push_back( pv );
        }

        /// @todo update函数里面需不需要访问mesh有关的部分
        std::sort( pv_list.begin(), pv_list.end(), var_contrast );      // 不太明白这里为什么需要排序
//        LOG(INFO) << "The size of pv_list is " << pv_list.size();
        updateVoxelMap( pv_list, m_max_voxel_size, m_max_layer, m_layer_init_size, m_max_points_size, m_min_eigen_value, m_feat_map );
//        LOG(INFO) << "Update the voxelMap";

        double vx_map_cost_time = omp_get_wtime();
        g_vx_map_frame_cost_time = ( vx_map_cost_time - g_LiDAR_frame_start_time ) * 1000.0;
        // cout << "vx_map_cost_time = " <<  g_vx_map_frame_cost_time << " ms" << endl;

        transformLidar( state.rot_end, state.pos_end, m_feats_undistort, world_lidar_full );

        // 打包数据成list | 主要是用于重建 mesh图的信息
        g_mutex_data_package_lock.lock();
        g_rec_mesh_data_package_list.emplace_back( world_lidar_full, Eigen::Quaterniond( state.rot_end ), state.pos_end, g_frame_idx );
        g_mutex_data_package_lock.unlock();

        open_log_file();
        if ( g_fp_lio_state != nullptr )
        {
            dump_lio_state_to_log( g_fp_lio_state );
        }
        g_frame_idx++;
    }

    // 保存一些点云信息
    if ( !m_use_new_map )
    {
        for ( int i = 0; i < m_feats_down_size; i++ )
        {
            /* transform to world frame */
            pointBodyToWorld( m_feats_down_body->points[ i ], m_feats_down_world->points[ i ] );
        }
        
        // add_to_offline_bin( state, m_Lidar_Measures.lidar_beg_time, m_feats_down_world );
        
#ifdef USE_ikdtree
#ifdef USE_ikdforest
        ikdforest.Add_Points( feats_down_world->points, lidar_end_time );
#else
        m_ikdtree.Add_Points( m_feats_down_world->points, true );
#endif
#endif
    }
}


// 自行设置的增量式地图
void Voxel_mapping::map_incremental()
{
    start_mesh_threads( m_meshing_maximum_thread_for_rec_mesh );
    if ( m_use_new_map )
    {
        while ( g_flag_pause )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        }

        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar( new pcl::PointCloud< pcl::PointXYZI > );
        pcl::PointCloud< pcl::PointXYZI >::Ptr world_lidar_full( new pcl::PointCloud< pcl::PointXYZI > );

        std::vector< Point_with_var > pv_list;

        transformLidar( state.rot_end, state.pos_end, m_feats_down_body, world_lidar );

        for ( size_t i = 0; i < world_lidar->size(); i++ )
        {
            Point_with_var pv;
            pv.m_point << world_lidar->points[ i ].x, world_lidar->points[ i ].y, world_lidar->points[ i ].z;

            /// @bug 输出协方差出现了问题 | 用原始公式计算的协方差一直是存在一些问题的(这里直接忽略计算)
//            M3D point_crossmat = m_cross_mat_list[ i ];
            M3D var;
            calcBodyVar( pv.m_point, m_dept_err, m_beam_err, var );
//            var = ( state.rot_end * m_extR ) * var * ( state.rot_end * m_extR ).transpose() +
//                  ( -point_crossmat ) * state.cov.block< 3, 3 >( 0, 0 ) * ( -point_crossmat ).transpose() + state.cov.block< 3, 3 >( 3, 3 );
            pv.m_var = var;
            pv_list.push_back( pv );
        }


        /// @todo update函数里面需不需要访问mesh有关的部分


        std::sort( pv_list.begin(), pv_list.end(), var_contrast );      // 不太明白这里为什么需要排序
        updateVoxelMap( pv_list, m_max_voxel_size, m_max_layer, m_layer_init_size, m_max_points_size, m_min_eigen_value, m_feat_map );
        LOG(INFO) << "[service_LiDAR_update] Update the voxelMap";

        double vx_map_cost_time = omp_get_wtime();
        g_vx_map_frame_cost_time = ( vx_map_cost_time - g_LiDAR_frame_start_time ) * 1000.0;
        // cout << "vx_map_cost_time = " <<  g_vx_map_frame_cost_time << " ms" << endl;

        transformLidar( state.rot_end, state.pos_end, m_feats_undistort, world_lidar_full );

        // 打包数据成list | 主要是用于重建 mesh图的信息
        g_mutex_data_package_lock.lock();
        g_rec_mesh_data_package_list.emplace_back( world_lidar_full, Eigen::Quaterniond( state.rot_end ), state.pos_end, g_frame_idx );
        g_mutex_data_package_lock.unlock();

        open_log_file();
        if ( g_fp_lio_state != nullptr )
        {
            dump_lio_state_to_log( g_fp_lio_state );
        }
        g_frame_idx++;
    }

}