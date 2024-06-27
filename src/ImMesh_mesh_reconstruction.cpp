/* 
This code is the implementation of our paper "ImMesh: An Immediate LiDAR Localization and Meshing Framework".

The source code of this package is released under GPLv2 license. We only allow it free for personal and academic usage. 

If you use any code of this repo in your academic research, please cite at least one of our papers:
[1] Lin, Jiarong, et al. "Immesh: An immediate lidar localization and meshing framework." IEEE Transactions on Robotics
   (T-RO 2023)
[2] Yuan, Chongjian, et al. "Efficient and probabilistic adaptive voxel mapping for accurate online lidar odometry."
    IEEE Robotics and Automation Letters (RA-L 2022)
[3] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored, LiDAR-Inertial-Visual tightly-coupled
    state Estimation and mapping package." IEEE International Conference on Robotics and Automation (ICRA 2022)

For commercial use, please contact me <ziv.lin.ljr@gmail.com> and Dr. Fu Zhang <fuzhang@hku.hk> to negotiate a 
different license.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include "voxel_mapping.hpp"
#include "meshing/mesh_rec_display.hpp"
#include "meshing/mesh_rec_geometry.hpp"
#include "tools/tools_thread_pool.hpp"

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

int        g_maximum_thread_for_rec_mesh;
std::mutex g_mutex_append_map;
std::mutex g_mutex_reconstruct_mesh;

extern double g_LiDAR_frame_start_time;
double        g_vx_map_frame_cost_time;
static double g_LiDAR_frame_avg_time;

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

    // g_eigen_vec_vec把每一个点都转换成为矩阵 () | 将一帧的所有点云 与 当前lidar的位置设置了映射关系
    for ( int i = 0; i < frame_pts->points.size(); i++ )
    {
        g_eigen_vec_vec[ frame_idx ].first.emplace_back( frame_pts->points[ i ].x, frame_pts->points[ i ].y, frame_pts->points[ i ].z,
                                                         frame_pts->points[ i ].intensity );
    }
    g_eigen_vec_vec[ frame_idx ].second = pose_vec;
    // g_eigen_vec_vec.push_back( std::make_pair( empty_vec, pose_vec ) );
    // TODO : add time tic toc


    // appending_pts_frame : 一帧frame中有多少点可以被加入到global map中 | append_point_step 使用自适应的步长(限制每一帧点云中应该有多少点可以被加入到地图中)
    int                 append_point_step = std::max( ( int ) 1, ( int ) std::round( frame_pts->points.size() / appending_pts_frame ) );
    Common_tools::Timer tim, tim_total;


    // g_map_rgb_pts_mesh 对应的为全局地图
    g_mutex_append_map.lock();
    g_map_rgb_pts_mesh.append_points_to_global_map( *frame_pts, frame_idx, nullptr, append_point_step );
    // 知道哪些voxel需要被处理(这个数据会跟临时定义的voxels_recent_visited变量进行数据互换，所以没有显示地让整个数据清空的操作)
    std::unordered_set< std::shared_ptr< RGB_Voxel > > voxels_recent_visited = g_map_rgb_pts_mesh.m_voxels_recent_visited;
    g_mutex_append_map.unlock();

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
                pts_in_voxels.push_back( g_map_rgb_pts_mesh.m_rgb_pts_vec[ p ] );
            }
           
            std::set< long > convex_hull_index, inner_pts_index;
            // mtx_triangle_lock.lock();

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
                correct_triangle_index( triangle_ptr, g_eigen_vec_vec[ frame_idx ].second.block( 4, 0, 3, 1 ), voxel->m_short_axis );
            }
            for ( auto triangle_ptr : existing_triangle )
            {
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
    LOG(INFO) << "[incremental_mesh_reconstruction]: The count of deleted triangle is "<< total_delete_triangle;

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
    LOG(INFO) << "[incremental_mesh_reconstruction]: The count of added triangle is "<< total_add_triangle;
    
    g_mutex_reconstruct_mesh.unlock();
   
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



/// @brief 新建/使用线程池,不断处理数据

void service_reconstruct_mesh()
{
    LOG(INFO) << "Starting the mesh thread";
    // 创建一个线程池
    if ( g_thread_pool_rec_mesh == nullptr )
    {
        // 不需要手动删除(线程池自己会进行销毁工作) —— 在线程池类的析构函数中进行数据的销毁
        g_thread_pool_rec_mesh = std::make_shared< Common_tools::ThreadPool >( g_maximum_thread_for_rec_mesh );
    }
    int drop_frame_num = 0;
    while ( 1 )
    {
            // 等待数据
            while ( g_rec_mesh_data_package_list.size() == 0 )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
            }

            // 这里上锁是因为service_lidar_update在另一个线程中会放入数据，而在这个线程要取出数据所以上锁
            g_mutex_data_package_lock.lock();

            // 维持数据量
            while ( g_rec_mesh_data_package_list.size() > 1e5 )
            {
                cout << "Drop mesh frame [" << g_rec_mesh_data_package_list.front().m_frame_idx;
                printf( "], total_drop = %d, all_frame = %d\r\n", drop_frame_num++, g_rec_mesh_data_package_list.front().m_frame_idx );
                g_rec_mesh_data_package_list.pop_front();
            }

            // 目前需要处理的数据量
            if ( g_rec_mesh_data_package_list.size() > 10 )
            {
                LOG(INFO) << "Poor real-time performance, current buffer size = " <<  g_rec_mesh_data_package_list.size();
//                cout << "Poor real-time performance, current buffer size = " << g_rec_mesh_data_package_list.size() << endl;
            }

            Rec_mesh_data_package data_pack_front = g_rec_mesh_data_package_list.front();
            g_rec_mesh_data_package_list.pop_front();
            g_mutex_data_package_lock.unlock();     // 数据读取好就可以放锁

            // ANCHOR - Comment follow line to disable meshing
            if ( g_enable_mesh_rec )
            {
                // 多线程的部分开始运行 | 给定的参数分别为 : 任务函数 + 一些参数
                /// @attention 这里说的是某一个空闲线程来接受这个任务而不是所有的线程都在执行这个函数
                g_thread_pool_rec_mesh->commit_task( incremental_mesh_reconstruction, data_pack_front.m_frame_pts, data_pack_front.m_pose_q,
                                                     data_pack_front.m_pose_t, data_pack_front.m_frame_idx );
            }

        std::this_thread::sleep_for( std::chrono::microseconds( 10 ) );
    }
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
//    start_mesh_threads( m_meshing_maximum_thread_for_rec_mesh );
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